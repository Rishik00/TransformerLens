"""Unit tests for PythiaArchitectureAdapter.

Tests cover:
- Adapter-specific config defaults
- Component mapping structure and HF module paths
- Interleaved GPT-NeoX/Pythia QKV split behavior
- Expected weight conversion keys
- setup_component_testing rotary embedding wiring
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.conversion_utils.conversion_steps.chain_tensor_conversion import (
    ChainTensorConversion,
)
from transformer_lens.conversion_utils.conversion_steps.rearrange_tensor_conversion import (
    RearrangeTensorConversion,
)
from transformer_lens.conversion_utils.conversion_steps.split_tensor_conversion import (
    SplitTensorConversion,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.generalized_components import (
    EmbeddingBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    ParallelBlockBridge,
    RotaryEmbeddingBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.supported_architectures.pythia import (
    PythiaArchitectureAdapter,
)


def _make_cfg(
    n_heads: int = 4,
    d_model: int = 64,
    n_layers: int = 2,
    d_mlp: int = 256,
    d_vocab: int = 100,
    n_ctx: int = 64,
) -> TransformerBridgeConfig:
    return TransformerBridgeConfig(
        d_model=d_model,
        d_head=d_model // n_heads,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_vocab=d_vocab,
        d_mlp=d_mlp,
        architecture="GPTNeoXForCausalLM",
    )


@pytest.fixture
def cfg() -> TransformerBridgeConfig:
    return _make_cfg()


@pytest.fixture
def adapter(cfg: TransformerBridgeConfig) -> PythiaArchitectureAdapter:
    return PythiaArchitectureAdapter(cfg)


class FakePythiaAttention(nn.Module):
    """Minimal GPT-NeoX/Pythia-style attention with interleaved per-head QKV."""

    def __init__(self, cfg: TransformerBridgeConfig) -> None:
        super().__init__()
        out_features = cfg.n_heads * 3 * cfg.d_head
        # GPT-NeoX/Pythia packs Q, K, V into one fused projection.
        self.query_key_value = nn.Linear(cfg.d_model, out_features, bias=True)
        self.dense = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=True)


def _fake_hf_model(rotary_emb: object) -> SimpleNamespace:
    return SimpleNamespace(gpt_neox=SimpleNamespace(rotary_emb=rotary_emb))


class DummyAttention:
    def __init__(self) -> None:
        self.rotary_emb = None

    def set_rotary_emb(self, rotary_emb: object) -> None:
        self.rotary_emb = rotary_emb


class DummyBlock:
    def __init__(self, has_attention: bool = True) -> None:
        if has_attention:
            self.attn = DummyAttention()


class DummyBridgeModel:
    def __init__(self, blocks: list[DummyBlock]) -> None:
        self.blocks = blocks


class TestPythiaAdapterConfig:
    def test_positional_embedding_type_is_rotary(self, adapter: PythiaArchitectureAdapter) -> None:
        assert adapter.cfg.positional_embedding_type == "rotary"

    def test_parallel_attn_mlp_is_true(self, adapter: PythiaArchitectureAdapter) -> None:
        assert adapter.cfg.parallel_attn_mlp is True

    def test_default_prepend_bos_is_false(self, adapter: PythiaArchitectureAdapter) -> None:
        assert adapter.cfg.default_prepend_bos is False


class TestPythiaComponentMapping:
    def test_top_level_keys(self, adapter: PythiaArchitectureAdapter) -> None:
        assert set(adapter.component_mapping.keys()) == {
            "embed",
            "rotary_emb",
            "blocks",
            "ln_final",
            "unembed",
        }

    def test_bridge_types(self, adapter: PythiaArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        blocks = mapping["blocks"]
        assert isinstance(mapping["embed"], EmbeddingBridge)
        assert isinstance(mapping["rotary_emb"], RotaryEmbeddingBridge)
        assert isinstance(blocks, ParallelBlockBridge)
        assert isinstance(mapping["ln_final"], NormalizationBridge)
        assert isinstance(mapping["unembed"], UnembeddingBridge)
        assert isinstance(blocks.submodules["attn"], JointQKVPositionEmbeddingsAttentionBridge)
        assert isinstance(blocks.submodules["mlp"], MLPBridge)

    def test_hf_paths(self, adapter: PythiaArchitectureAdapter) -> None:
        mapping = adapter.component_mapping
        blocks = mapping["blocks"]
        attn = blocks.submodules["attn"]
        mlp = blocks.submodules["mlp"]

        assert mapping["embed"].name == "gpt_neox.embed_in"
        assert mapping["rotary_emb"].name == "gpt_neox.rotary_emb"
        assert blocks.name == "gpt_neox.layers"
        assert mapping["ln_final"].name == "gpt_neox.final_layer_norm"
        assert mapping["unembed"].name == "embed_out"
        assert blocks.submodules["ln1"].name == "input_layernorm"
        assert blocks.submodules["ln2"].name == "post_attention_layernorm"
        assert attn.name == "attention"
        assert attn.submodules["qkv"].name == "query_key_value"
        assert attn.submodules["o"].name == "dense"
        assert mlp.submodules["in"].name == "dense_h_to_4h"
        assert mlp.submodules["out"].name == "dense_4h_to_h"

    def test_linear_submodule_bridge_types(self, adapter: PythiaArchitectureAdapter) -> None:
        blocks = adapter.component_mapping["blocks"]
        attn = blocks.submodules["attn"]
        mlp = blocks.submodules["mlp"]
        assert isinstance(attn.submodules["qkv"], LinearBridge)
        assert isinstance(attn.submodules["o"], LinearBridge)
        assert isinstance(mlp.submodules["in"], LinearBridge)
        assert isinstance(mlp.submodules["out"], LinearBridge)


class TestPythiaSplitQKV:
    def test_split_qkv_matrix_recovers_interleaved_weights(
        self, adapter: PythiaArchitectureAdapter
    ) -> None:
        fake_attn = FakePythiaAttention(adapter.cfg)
        total_rows = adapter.cfg.n_heads * 3 * adapter.cfg.d_head

        fake_attn.query_key_value.weight.data.copy_(
            torch.arange(total_rows * adapter.cfg.d_model, dtype=torch.float32).view(
                total_rows, adapter.cfg.d_model
            )
        )
        fake_attn.query_key_value.bias.data.copy_(torch.arange(total_rows, dtype=torch.float32))

        q, k, v = adapter.split_qkv_matrix(fake_attn)

        expected = fake_attn.query_key_value.weight.view(
            adapter.cfg.n_heads, 3 * adapter.cfg.d_head, adapter.cfg.d_model
        )
        expected_q = expected[:, : adapter.cfg.d_head, :].reshape(
            adapter.cfg.n_heads * adapter.cfg.d_head, adapter.cfg.d_model
        )
        expected_k = expected[:, adapter.cfg.d_head : 2 * adapter.cfg.d_head, :].reshape(
            adapter.cfg.n_heads * adapter.cfg.d_head, adapter.cfg.d_model
        )
        expected_v = expected[:, 2 * adapter.cfg.d_head :, :].reshape(
            adapter.cfg.n_heads * adapter.cfg.d_head, adapter.cfg.d_model
        )

        assert torch.equal(q.weight, expected_q)
        assert torch.equal(k.weight, expected_k)
        assert torch.equal(v.weight, expected_v)

    def test_split_qkv_matrix_recovers_interleaved_bias(
        self, adapter: PythiaArchitectureAdapter
    ) -> None:
        fake_attn = FakePythiaAttention(adapter.cfg)
        total_rows = adapter.cfg.n_heads * 3 * adapter.cfg.d_head
        fake_attn.query_key_value.bias.data.copy_(torch.arange(total_rows, dtype=torch.float32))

        q, k, v = adapter.split_qkv_matrix(fake_attn)
        expected = fake_attn.query_key_value.bias.view(adapter.cfg.n_heads, 3 * adapter.cfg.d_head)

        assert torch.equal(q.bias, expected[:, : adapter.cfg.d_head].reshape(-1))
        assert torch.equal(
            k.bias, expected[:, adapter.cfg.d_head : 2 * adapter.cfg.d_head].reshape(-1)
        )
        assert torch.equal(v.bias, expected[:, 2 * adapter.cfg.d_head :].reshape(-1))


class TestPythiaAttentionHookShapes:
    N_HEADS = 4
    D_MODEL = 64
    D_HEAD = D_MODEL // N_HEADS
    BATCH = 2
    SEQ = 8

    @pytest.fixture
    def adapter(self) -> PythiaArchitectureAdapter:
        return PythiaArchitectureAdapter(_make_cfg(n_heads=self.N_HEADS, d_model=self.D_MODEL))

    @pytest.fixture
    def wired_attn_bridge(
        self, adapter: PythiaArchitectureAdapter
    ) -> JointQKVPositionEmbeddingsAttentionBridge:
        fake_attn = FakePythiaAttention(adapter.cfg)
        attn_bridge = adapter.component_mapping["blocks"].submodules["attn"]
        assert isinstance(attn_bridge, JointQKVPositionEmbeddingsAttentionBridge)
        attn_bridge.set_original_component(fake_attn)
        # A full TransformerBridge build would register these children for us.
        # Wire them by hand here so the test can execute the bridge forward path.
        for name, submodule in {
            "q": attn_bridge.q,
            "k": attn_bridge.k,
            "v": attn_bridge.v,
            "o": attn_bridge.submodules["o"],
        }.items():
            if name == "o":
                submodule.set_original_component(fake_attn.dense)
            attn_bridge.add_module(name, submodule)
        attn_bridge.setup_hook_compatibility()
        return attn_bridge

    def _run_and_capture(
        self, attn_bridge: JointQKVPositionEmbeddingsAttentionBridge
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        captured = {}

        def _capture(name: str):
            def _hook(x, hook):
                captured[name] = x.detach()
                return x

            return _hook

        attn_bridge.q.hook_out.add_hook(_capture("q"))
        attn_bridge.k.hook_out.add_hook(_capture("k"))
        attn_bridge.v.hook_out.add_hook(_capture("v"))

        hidden = torch.randn(self.BATCH, self.SEQ, self.D_MODEL)
        # Identity RoPE inputs keep the test focused on bridge reshaping.
        cos = torch.ones(1, self.SEQ, self.D_HEAD)
        sin = torch.zeros(1, self.SEQ, self.D_HEAD)
        out = attn_bridge(hidden, position_embeddings=(cos, sin))
        out_tensor = out[0] if isinstance(out, tuple) else out

        return captured["q"], captured["k"], captured["v"], out_tensor

    def test_hook_q_shape(
        self, wired_attn_bridge: JointQKVPositionEmbeddingsAttentionBridge
    ) -> None:
        q, _, _, _ = self._run_and_capture(wired_attn_bridge)
        assert q.shape == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD)

    def test_hook_k_shape(
        self, wired_attn_bridge: JointQKVPositionEmbeddingsAttentionBridge
    ) -> None:
        _, k, _, _ = self._run_and_capture(wired_attn_bridge)
        assert k.shape == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD)

    def test_hook_v_shape(
        self, wired_attn_bridge: JointQKVPositionEmbeddingsAttentionBridge
    ) -> None:
        _, _, v, _ = self._run_and_capture(wired_attn_bridge)
        assert v.shape == (self.BATCH, self.SEQ, self.N_HEADS, self.D_HEAD)

    def test_attn_output_shape(
        self, wired_attn_bridge: JointQKVPositionEmbeddingsAttentionBridge
    ) -> None:
        _, _, _, out = self._run_and_capture(wired_attn_bridge)
        assert out.shape == (self.BATCH, self.SEQ, self.D_MODEL)


class TestPythiaWeightConversions:
    def test_expected_conversion_keys_present(self, adapter: PythiaArchitectureAdapter) -> None:
        assert set(adapter.weight_processing_conversions.keys()) == {
            "blocks.{i}.attn.q",
            "blocks.{i}.attn.k",
            "blocks.{i}.attn.v",
            "blocks.{i}.attn.b_Q",
            "blocks.{i}.attn.b_K",
            "blocks.{i}.attn.b_V",
            "blocks.{i}.attn.o",
        }

    def test_qkv_weight_conversions_are_split_then_reshaped(
        self, adapter: PythiaArchitectureAdapter
    ) -> None:
        expected_indices = {
            "blocks.{i}.attn.q": 0,
            "blocks.{i}.attn.k": 1,
            "blocks.{i}.attn.v": 2,
        }

        for key, split_idx in expected_indices.items():
            conv = adapter.weight_processing_conversions[key]
            assert isinstance(conv, ParamProcessingConversion)
            assert conv.source_key == "gpt_neox.layers.{i}.attention.query_key_value.weight"
            assert isinstance(conv.tensor_conversion, ChainTensorConversion)
            split, rearrange = conv.tensor_conversion.conversions
            assert isinstance(split, SplitTensorConversion)
            assert split.index == split_idx
            assert split.num_splits == 3
            assert split.dim == 0
            assert isinstance(rearrange, RearrangeTensorConversion)
            assert rearrange.pattern == "(head d_head) d_model -> head d_model d_head"
            assert rearrange.axes_lengths["head"] == adapter.cfg.n_heads
            assert rearrange.axes_lengths["d_head"] == adapter.cfg.d_head

    def test_qkv_bias_conversions_are_split_then_reshaped(
        self, adapter: PythiaArchitectureAdapter
    ) -> None:
        expected_indices = {
            "blocks.{i}.attn.b_Q": 0,
            "blocks.{i}.attn.b_K": 1,
            "blocks.{i}.attn.b_V": 2,
        }

        for key, split_idx in expected_indices.items():
            conv = adapter.weight_processing_conversions[key]
            assert isinstance(conv, ParamProcessingConversion)
            assert conv.source_key == "gpt_neox.layers.{i}.attention.query_key_value.bias"
            assert isinstance(conv.tensor_conversion, ChainTensorConversion)
            split, rearrange = conv.tensor_conversion.conversions
            assert isinstance(split, SplitTensorConversion)
            assert split.index == split_idx
            assert split.num_splits == 3
            assert split.dim == 0
            assert isinstance(rearrange, RearrangeTensorConversion)
            assert rearrange.pattern == "(head d_head) -> head d_head"
            assert rearrange.axes_lengths["head"] == adapter.cfg.n_heads

    def test_output_weight_conversion_uses_head_reshaping(
        self, adapter: PythiaArchitectureAdapter
    ) -> None:
        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o"]
        assert isinstance(conv, ParamProcessingConversion)
        assert conv.source_key == "gpt_neox.layers.{i}.attention.dense.weight"
        assert isinstance(conv.tensor_conversion, RearrangeTensorConversion)
        assert conv.tensor_conversion.pattern == "d_model (head d_head) -> head d_head d_model"
        assert conv.tensor_conversion.axes_lengths["head"] == adapter.cfg.n_heads
        assert conv.tensor_conversion.axes_lengths["d_head"] == adapter.cfg.d_head

    @pytest.mark.xfail(
        strict=True,
        reason="Current Pythia weight conversions chunk fused QKV contiguously instead of per-head interleaving.",
    )
    def test_qkv_weight_conversions_match_interleaved_layout(
        self, adapter: PythiaArchitectureAdapter
    ) -> None:
        total_rows = adapter.cfg.n_heads * 3 * adapter.cfg.d_head
        fused_weight = torch.arange(total_rows * adapter.cfg.d_model, dtype=torch.float32).view(
            total_rows, adapter.cfg.d_model
        )
        # Real Pythia layout is [Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...].
        reshaped = fused_weight.view(
            adapter.cfg.n_heads, 3 * adapter.cfg.d_head, adapter.cfg.d_model
        )

        expected = {
            "blocks.{i}.attn.q": reshaped[:, : adapter.cfg.d_head, :].permute(0, 2, 1),
            "blocks.{i}.attn.k": reshaped[
                :, adapter.cfg.d_head : 2 * adapter.cfg.d_head, :
            ].permute(0, 2, 1),
            "blocks.{i}.attn.v": reshaped[:, 2 * adapter.cfg.d_head :, :].permute(0, 2, 1),
        }

        for key, expected_value in expected.items():
            conv = adapter.weight_processing_conversions[key]
            actual = conv.tensor_conversion.handle_conversion(fused_weight)
            assert torch.equal(actual, expected_value)

    @pytest.mark.xfail(
        strict=True,
        reason="Current Pythia bias conversions chunk fused QKV contiguously instead of per-head interleaving.",
    )
    def test_qkv_bias_conversions_match_interleaved_layout(
        self, adapter: PythiaArchitectureAdapter
    ) -> None:
        total_rows = adapter.cfg.n_heads * 3 * adapter.cfg.d_head
        fused_bias = torch.arange(total_rows, dtype=torch.float32)
        # Bias follows the same per-head interleaving as the fused weight matrix.
        reshaped = fused_bias.view(adapter.cfg.n_heads, 3 * adapter.cfg.d_head)

        expected = {
            "blocks.{i}.attn.b_Q": reshaped[:, : adapter.cfg.d_head],
            "blocks.{i}.attn.b_K": reshaped[:, adapter.cfg.d_head : 2 * adapter.cfg.d_head],
            "blocks.{i}.attn.b_V": reshaped[:, 2 * adapter.cfg.d_head :],
        }

        for key, expected_value in expected.items():
            conv = adapter.weight_processing_conversions[key]
            actual = conv.tensor_conversion.handle_conversion(fused_bias)
            assert torch.equal(actual, expected_value)

    def test_output_weight_conversion_matches_expected_shape_and_values(
        self, adapter: PythiaArchitectureAdapter
    ) -> None:
        dense_weight = torch.arange(
            adapter.cfg.d_model * adapter.cfg.n_heads * adapter.cfg.d_head, dtype=torch.float32
        ).view(adapter.cfg.d_model, adapter.cfg.n_heads * adapter.cfg.d_head)

        conv = adapter.weight_processing_conversions["blocks.{i}.attn.o"]
        actual = conv.tensor_conversion.handle_conversion(dense_weight)
        expected = dense_weight.view(
            adapter.cfg.d_model, adapter.cfg.n_heads, adapter.cfg.d_head
        ).permute(1, 2, 0)

        assert torch.equal(actual, expected)


class TestPythiaSetupComponentTesting:
    def test_sets_rotary_emb_on_template_attention(
        self, adapter: PythiaArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        attn_template = adapter.get_generalized_component("blocks.0.attn")
        assert isinstance(attn_template, JointQKVPositionEmbeddingsAttentionBridge)
        assert attn_template._rotary_emb is None

        adapter.setup_component_testing(_fake_hf_model(rotary_emb))

        assert attn_template._rotary_emb is rotary_emb

    def test_sets_rotary_emb_on_each_bridge_model_attention(
        self, adapter: PythiaArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(), DummyBlock()])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        for block in bridge_model.blocks:
            assert block.attn.rotary_emb is rotary_emb

    def test_skips_bridge_blocks_without_attention(
        self, adapter: PythiaArchitectureAdapter
    ) -> None:
        rotary_emb = object()
        bridge_model = DummyBridgeModel([DummyBlock(), DummyBlock(has_attention=False)])

        adapter.setup_component_testing(_fake_hf_model(rotary_emb), bridge_model=bridge_model)

        assert bridge_model.blocks[0].attn.rotary_emb is rotary_emb


class TestPythiaFactoryRegistration:
    def test_factory_key_present(self) -> None:
        assert "GPTNeoXForCausalLM" in SUPPORTED_ARCHITECTURES

    @pytest.mark.xfail(
        strict=True,
        reason="Factory still routes GPTNeoXForCausalLM through the generic NeoX adapter.",
    )
    def test_factory_maps_architecture_to_pythia_adapter(self) -> None:
        # This is the desired end state once Pythia gets its own selection path.
        assert SUPPORTED_ARCHITECTURES["GPTNeoXForCausalLM"] is PythiaArchitectureAdapter

    @pytest.mark.xfail(
        strict=True,
        reason="Factory still routes GPTNeoXForCausalLM through the generic NeoX adapter.",
    )
    def test_factory_returns_pythia_instance(self) -> None:
        cfg = _make_cfg()
        adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
        assert isinstance(adapter, PythiaArchitectureAdapter)
