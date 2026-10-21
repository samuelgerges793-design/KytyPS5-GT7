#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "common/assert.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {
namespace {
thread_local uint32_t g_last_waterfall_guards = 0;
thread_local uint32_t g_loop_cap_override    = 0;
} // namespace

uint32_t LastWaterfallLoopGuards() {
	return g_last_waterfall_guards;
}

void SetLoopCapOverride(uint32_t cap) {
	g_loop_cap_override = cap;
}

uint32_t LoopCapFor(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> list;
		if (const char* env = std::getenv("KYTY_LOOP_CAP_SHADERS"); env != nullptr) {
			const std::string text(env);
			size_t            start = 0;
			while (start < text.size()) {
				const auto end = std::min(text.find(',', start), text.size());
				if (end > start) list.push_back(std::stoull(text.substr(start, end - start), nullptr, 16));
				start = end + 1;
			}
		}
		return list;
	}();
	static const uint32_t cap = [] {
		const char* env = std::getenv("KYTY_LOOP_CAP");
		return env != nullptr ? static_cast<uint32_t>(std::strtoul(env, nullptr, 10)) : 64u;
	}();
	return std::ranges::find(hashes, shader_hash) != hashes.end() ? cap : 0u;
}
} // namespace Libs::Graphics::ShaderRecompiler

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

void EmitKillIfBoolFalse(EmitterState& state, uint32_t active) {
	const auto kill_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	const auto inactive    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalNot, TypeBool(state), inactive, active);
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, inactive, kill_label, merge_label);
	EmitLabel(state, kill_label);
	state.builder.AddFunction(spv::OpKill);
	EmitLabel(state, merge_label);
}

void EmitKillIfPixelValidMaskInactive(EmitterState& state) {
	if (state.pixel_valid_mask_variable == 0) {
		return;
	}

	const auto mask_value = state.builder.AllocateId();
	const auto active     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), mask_value,
	                          state.pixel_valid_mask_variable);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), active, mask_value,
	                          ConstantU32(state, 0));
	EmitKillIfBoolFalse(state, active);
}

uint32_t SpillPointerType(ValueEmitContext& ctx, IR::Type type) {
	const auto value_type = TypeId(ctx.state, type);
	return value_type == 0 ? 0 : TypePointer(ctx.state, spv::StorageClassFunction, value_type);
}

struct DeferredPhiPatch {
	DeferredPhi     phi;
	const IR::Inst* instruction = nullptr;
	uint32_t        half        = 0;
};

struct StructuredFunctionState {
	std::unordered_map<const IR::Block*, uint32_t> block_exit_labels;
	std::vector<DeferredPhiPatch>                  deferred_phis;
};

struct DispatcherFunctionState {
	std::array<std::unordered_map<const IR::Inst*, uint32_t>, 2> spills;
	uint32_t                                      header_label       = 0;
	uint32_t                                      select_label       = 0;
	uint32_t                                      after_switch_label = 0;
	uint32_t                                      continue_label     = 0;
	uint32_t                                      merge_label        = 0;
};

void StoreDispatcherPhiEdge(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                            const IR::Block* from, const IR::Block* to) {
	if (to == nullptr) {
		return;
	}
	for (const auto& phi: *to) {
		if (phi.GetOpcode() != IR::ValueOpcode::Phi) {
			break;
		}
		for (size_t index = 0; index < phi.NumArgs(); index++) {
			if (phi.PhiBlock(index) == from) {
				ctx.state.builder.AddFunction(spv::OpStore, dispatcher.spills[ctx.half].at(&phi),
				                              ctx.Def(phi.Arg(index)));
				break;
			}
		}
	}
}

void EmitReturn(ValueEmitContext& ctx) {
	EmitKillIfPixelValidMaskInactive(ctx.state);
	if (ctx.state.object_primitive_id_variable != 0) {
		auto& state = ctx.state;
		const auto primitive = EmitGetBuiltin(
		    ctx, IR::Value {static_cast<uint32_t>(IR::StageInputKind::PrimitiveId)}, IR::Value {0u});
		const auto object = EmitPushDataLoad(state, ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpStore, state.object_primitive_id_variable,
		                          EmitAddU32(state, object, primitive));
	}
	ctx.state.builder.AddFunction(spv::OpReturn);
}

void EmitStructuredTerminator(ValueEmitContext& ctx, const IR::Block* block) {
	const auto& term       = block->terminator;
	const auto  emit_merge = [&]() {
		if (term.loop_header) {
			const auto* merge = term.merge_block;
			const auto* cont  = term.continue_block;
			if (merge != nullptr && cont != nullptr) {
				ctx.state.builder.AddFunction(spv::OpLoopMerge, merge->Definition(),
				                              cont->Definition(), spv::LoopControlMaskNone);
			}
		} else if (term.kind == CFG::TerminatorKind::ConditionalBranch &&
		           term.merge_block != nullptr) {
			ctx.state.builder.AddFunction(spv::OpSelectionMerge, term.merge_block->Definition(),
			                              spv::SelectionControlMaskNone);
		}
	};

	switch (term.kind) {
		case CFG::TerminatorKind::Branch: {
			const auto* target = term.true_block;
			if (target == nullptr) {
				EmitReturn(ctx);
				return;
			}
			emit_merge();
			ctx.state.builder.AddFunction(spv::OpBranch, target->Definition());
			return;
		}
		case CFG::TerminatorKind::ConditionalBranch: {
			const auto* true_block  = term.true_block;
			const auto* false_block = term.false_block;
			if (true_block == nullptr || false_block == nullptr || block->condition.IsEmpty()) {
				EmitReturn(ctx);
				return;
			}
			auto condition = ctx.Def(block->condition);
			if (const auto guard = ctx.state.loop_watchdogs.find(block);
			    guard != ctx.state.loop_watchdogs.end()) {
				// Leave the loop through its existing exit once the counter passes the limit.
				auto&      builder = ctx.state.builder;
				const auto loaded  = builder.AllocateId();
				const auto counted = builder.AllocateId();
				const auto over    = builder.AllocateId();
				const auto guarded = builder.AllocateId();
				builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), loaded, guard->second);
				builder.AddFunction(spv::OpIAdd, TypeU32(ctx.state), counted, loaded,
				                    ConstantU32(ctx.state, 1));
				builder.AddFunction(spv::OpStore, guard->second, counted);
				// Graphics stages run per pixel or vertex, so a lower cap keeps a runaway loop
				// within the GPU timeout.
				auto limit = ctx.state.loop_watchdog_waterfalls.contains(block) ? 256u
				             : ctx.state.program.stage == ShaderType::Compute ? 1u << 16u
				                                                              : 1u << 10u;
				if (g_loop_cap_override != 0) limit = std::min(limit, g_loop_cap_override);
				builder.AddFunction(spv::OpUGreaterThan, TypeBool(ctx.state), over, counted,
				                    ConstantU32(ctx.state, limit));
				// The exit is the merge target: the header's own merge, for a back-edge block the
				// target that is not a loop header, or for a break the loop's merge block.
				const auto brk = ctx.state.loop_watchdog_breaks.find(block);
				const bool exit_is_true =
				    brk != ctx.state.loop_watchdog_breaks.end() ? brk->second
				    : term.loop_header ? term.merge_block == true_block
				                       : false_block->terminator.loop_header &&
				                             false_block->terminator.merge_block == true_block;
				if (exit_is_true) {
					builder.AddFunction(spv::OpLogicalOr, TypeBool(ctx.state), guarded, condition,
					                    over);
				} else {
					const auto stay = builder.AllocateId();
					builder.AddFunction(spv::OpLogicalNot, TypeBool(ctx.state), stay, over);
					builder.AddFunction(spv::OpLogicalAnd, TypeBool(ctx.state), guarded, condition,
					                    stay);
				}
				condition = guarded;
			}
			emit_merge();
			ctx.state.builder.AddFunction(spv::OpBranchConditional, condition,
			                              true_block->Definition(), false_block->Definition());
			return;
		}
		default: EmitReturn(ctx); return;
	}
}

void EmitDispatcherTarget(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                          const IR::Block* from, const IR::Block* block) {
	if (block != nullptr) {
		StoreDispatcherPhiEdge(ctx, dispatcher, from, block);
		if (ctx.other_half != nullptr) {
			StoreDispatcherPhiEdge(*ctx.other_half, dispatcher, from, block);
		}
	}
}

uint32_t EmitDispatcherNextPc(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                              const IR::Block* block) {
	const auto& term = block->terminator;
	switch (term.kind) {
		case CFG::TerminatorKind::Branch:
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			return ConstantU32(ctx.state, term.true_block->id);
		case CFG::TerminatorKind::ConditionalBranch: {
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			EmitDispatcherTarget(ctx, dispatcher, block, term.false_block);
			const auto selected = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), selected,
			                              ctx.Def(block->condition),
			                              ConstantU32(ctx.state, term.true_block->id),
			                              ConstantU32(ctx.state, term.false_block->id));
			return selected;
		}
		case CFG::TerminatorKind::IndirectBranch: {
			for (const auto* target: block->ImmSuccessors()) {
				EmitDispatcherTarget(ctx, dispatcher, block, target);
			}
			uint32_t selected = ConstantU32(ctx.state, UINT32_MAX);
			if (!block->indirect_target.IsEmpty()) {
				const auto selector = ctx.Def(block->indirect_target);
				for (const auto& branch: term.cases) {
					const auto match = ctx.state.builder.AllocateId();
					const auto next  = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpIEqual, TypeBool(ctx.state), match,
					                              selector, ConstantU32(ctx.state, branch.value));
					ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), next, match,
					                              ConstantU32(ctx.state, branch.target->id),
					                              selected);
					selected = next;
				}
			}
			return selected;
		}
		default: return ConstantU32(ctx.state, UINT32_MAX);
	}
}

template <typename T>
decltype(auto) Arg(ValueEmitContext& ctx, const IR::Inst& inst, size_t index) {
	if constexpr (std::is_same_v<T, const IR::Inst&>) {
		return inst;
	} else if constexpr (std::is_same_v<T, IR::Value>) {
		return inst.Arg(index);
	} else if constexpr (std::is_same_v<T, IR::ScalarReg>) {
		return inst.Arg(index).ScalarRegister();
	} else {
		static_assert(std::is_same_v<T, uint32_t>);
		return ctx.Def(inst.Arg(index));
	}
}

template <typename Context, typename Return, typename... Args>
void Invoke(Return (*emit)(Context&, Args...), ValueEmitContext& ctx, const IR::Inst& inst) {
	// A full instruction keeps metadata and predicated/lane operand loads lazy.
	static_assert(std::is_same_v<Context, ValueEmitContext> ||
	              std::is_same_v<Context, EmitterState>);
	auto& context = [&]() -> Context& {
		if constexpr (std::is_same_v<Context, EmitterState>)
			return ctx.state;
		else
			return ctx;
	}();
	constexpr bool has_inst = (std::is_same_v<Args, const IR::Inst&> || ...);
	[&]<size_t... I>(std::index_sequence<I...>) {
		static_assert(((!std::is_same_v<Args, const IR::Inst&> || I == 0) && ...));
		const auto call = [&] {
			return emit(context, Arg<Args>(ctx, inst, I - (has_inst && I != 0))...);
		};
		if constexpr (std::is_void_v<Return>) {
			call();
		} else {
			static_assert(std::is_same_v<Return, uint32_t>);
			ctx.Define(inst, call());
		}
	}(std::index_sequence_for<Args...> {});
}

void EmitDirectInstruction(ValueEmitContext& ctx, const IR::Inst& inst) {
	using Op = IR::ValueOpcode;
	const auto op = inst.GetOpcode();
	if (ctx.half != 0 &&
	    (op == Op::Ballot || op == Op::ReadFirstLane || op == Op::SubgroupUMinU32 ||
	     op == Op::ReadConstBuffer ||
	     (op == Op::LoadAddressU32 && ctx.Memory(inst).kind == IR::ResourceKind::ScalarAddress))) {
		// Whole-wave results and SMEM loads (SGPR operands, no EXEC) are identical in both halves.
		ctx.Define(inst, ctx.other_half->Result(inst));
		return;
	}
	switch (op) {
#define VALUE_OPCODE(name, ...)                                                                    \
	case IR::ValueOpcode::name: return Invoke(Emit##name, ctx, inst);
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.inc"
#undef VALUE_OPCODE
		default: ctx.Fail(inst, "has no direct SPIR-V emitter");
	}
}

void EmitStructuredInstruction(ValueEmitContext& ctx, StructuredFunctionState& structured,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0 || inst.NumArgs() == 0) {
			ctx.Fail(inst, "has no native SPIR-V representation");
		}
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			const auto* predecessor = inst.PhiBlock(index);
			if (predecessor == nullptr || predecessor->Definition() == 0) {
				ctx.Fail(inst, "has a predecessor outside the structured function");
			}
		}
		structured.deferred_phis.push_back(
		    {ctx.state.builder.AddDeferredPhi(type, ctx.Result(inst), inst.NumArgs()), &inst,
		     ctx.half});
		return;
	}
	EmitDirectInstruction(ctx, inst);
}

void EmitDispatcherInstruction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0) {
			ctx.Fail(inst, "cannot be loaded by the dispatcher");
		}
		ctx.state.builder.AddFunction(spv::OpLoad, type, ctx.Result(inst),
		                              dispatcher.spills[ctx.half].at(&inst));
		return;
	}
	EmitDirectInstruction(ctx, inst);
	if (const auto found = dispatcher.spills[ctx.half].find(&inst);
	    found != dispatcher.spills[ctx.half].end()) {
		ctx.state.builder.AddFunction(spv::OpStore, found->second,
		                              ctx.Def(IR::Value(const_cast<IR::Inst*>(&inst))));
	}
}

template <typename EmitInstruction>
void EmitBlock(ValueEmitContext& ctx, const IR::Block* block, EmitInstruction&& emit_instruction) {
	ctx.state.current_block = block;
	EmitLabel(ctx.state, block->Definition());
	bool emitted_non_phi = false;
	for (const auto& inst: *block) {
		if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
			if (emitted_non_phi) {
				ctx.Fail(inst, "appears after a non-Phi instruction");
			}
		} else {
			emitted_non_phi = true;
		}
		for (uint32_t half = 0; half < ctx.state.lane_count; half++) {
			auto& lane          = half == 0 ? ctx : *ctx.other_half;
			ctx.state.lane_half = half;
			if (half == 0 || (inst.GetOpcode() != IR::ValueOpcode::Barrier &&
			                  inst.GetOpcode() != IR::ValueOpcode::StoreCompletion &&
			                  inst.GetOpcode() != IR::ValueOpcode::DataCacheInvalidate &&
			                  inst.GetOpcode() != IR::ValueOpcode::MeshAllocate)) {
				emit_instruction(lane, inst);
			}
		}
		ctx.state.lane_half = 0;
	}
	if (const auto reset = ctx.state.loop_watchdog_resets.find(block);
	    reset != ctx.state.loop_watchdog_resets.end()) {
		for (const auto counter: reset->second) {
			ctx.state.builder.AddFunction(spv::OpStore, counter, ConstantU32(ctx.state, 0));
		}
	}
}

void PatchStructuredPhis(ValueEmitContext& ctx, StructuredFunctionState& structured) {
	for (const auto& deferred: structured.deferred_phis) {
		auto& lane = deferred.half == 0 ? ctx : *ctx.other_half;
		for (size_t index = 0; index < deferred.instruction->NumArgs(); index++) {
			const auto* predecessor = deferred.instruction->PhiBlock(index);
			const auto  found       = structured.block_exit_labels.find(predecessor);
			if (found == structured.block_exit_labels.end()) {
				ctx.Fail(*deferred.instruction, "has a predecessor that was not emitted");
			}
			ctx.state.builder.PatchDeferredPhi(
			    deferred.phi, index, lane.Def(deferred.instruction->Arg(index)), found->second);
		}
	}
}

void EmitStructuredFunction(ValueEmitContext& ctx) {
	const auto& program = ctx.state.program;
	StructuredFunctionState structured;
	ctx.state.builder.AddFunction(spv::OpBranch, program.blocks.front()->Definition());
	for (size_t index = 0; index < program.blocks.size(); index++) {
		const auto* block = program.blocks[index];
		EmitBlock(ctx, block, [&](ValueEmitContext& lane, const IR::Inst& inst) {
			EmitStructuredInstruction(lane, structured, inst);
		});
		structured.block_exit_labels.emplace(block, ctx.state.current_label);
		EmitStructuredTerminator(ctx, block);
	}
	PatchStructuredPhis(ctx, structured);
}

void EmitDispatcherFunction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher) {
	auto&       state = ctx.state;
	const auto* entry = state.program.blocks.front();
	state.builder.AddFunction(spv::OpBranch, entry->Definition());
	EmitBlock(ctx, entry, [&](ValueEmitContext& lane, const IR::Inst& inst) {
		EmitDispatcherInstruction(lane, dispatcher, inst);
	});
	const auto initial_pc     = EmitDispatcherNextPc(ctx, dispatcher, entry);
	const auto initial_parent = state.current_label;
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);

	state.current_block = nullptr;
	EmitLabel(state, dispatcher.header_label);
	const auto pc      = state.builder.AllocateId();
	const auto next_pc = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), pc, initial_pc, initial_parent, next_pc,
	                          dispatcher.continue_label);
	// Experimental watchdog: count executed guest blocks and leave once a runaway loop passes
	// 2^21 of them, instead of hanging the GPU (KYTY_NO_LOOP_WATCHDOG disables).
	static const bool watchdog = std::getenv("KYTY_NO_LOOP_WATCHDOG") == nullptr;
	const auto steps      = state.builder.AllocateId();
	const auto next_steps = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), steps, ConstantU32(state, 0),
	                          initial_parent, next_steps, dispatcher.continue_label);
	state.builder.AddFunction(spv::OpIAdd, TypeU32(state), next_steps, steps,
	                          ConstantU32(state, 1));
	const auto finished = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), finished, pc,
	                          ConstantU32(ctx.state, UINT32_MAX));
	auto done = finished;
	if (watchdog) {
		const auto runaway = state.builder.AllocateId();
		done               = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpUGreaterThan, TypeBool(state), runaway, next_steps,
		                          ConstantU32(state, 1u << 21u));
		state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), done, finished, runaway);
	}
	state.builder.AddFunction(spv::OpLoopMerge, dispatcher.merge_label, dispatcher.continue_label,
	                          spv::LoopControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, done, dispatcher.merge_label,
	                          dispatcher.select_label);

	EmitLabel(state, dispatcher.select_label);
	state.builder.AddFunction(spv::OpSelectionMerge, dispatcher.after_switch_label,
	                          spv::SelectionControlMaskNone);
	std::vector<uint32_t> words {spv::OpSwitch, pc, dispatcher.after_switch_label};
	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		words.push_back(state.program.blocks[index]->id);
		words.push_back(state.program.blocks[index]->Definition());
	}
	state.builder.AddFunction(words);
	std::vector<uint32_t> next_pc_words {spv::OpPhi, TypeU32(state), next_pc,
	                                     ConstantU32(state, UINT32_MAX), dispatcher.select_label};

	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		EmitBlock(ctx, state.program.blocks[index],
		          [&](ValueEmitContext& lane, const IR::Inst& inst) {
			          EmitDispatcherInstruction(lane, dispatcher, inst);
		          });
		const auto selected = EmitDispatcherNextPc(ctx, dispatcher, state.program.blocks[index]);
		next_pc_words.push_back(selected);
		next_pc_words.push_back(state.current_label);
		state.builder.AddFunction(spv::OpBranch, dispatcher.after_switch_label);
	}
	state.current_block = nullptr;
	EmitLabel(state, dispatcher.after_switch_label);
	state.builder.AddFunction(next_pc_words);
	state.builder.AddFunction(spv::OpBranch, dispatcher.continue_label);
	EmitLabel(state, dispatcher.continue_label);
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);
	EmitLabel(state, dispatcher.merge_label);
	EmitReturn(ctx);
}

} // namespace

uint32_t TypeId(EmitterState& state, IR::Type type) {
	switch (type) {
		case IR::Type::U1: return TypeBool(state);
		case IR::Type::U8:
		case IR::Type::U16:
		case IR::Type::U32:
		case IR::Type::F16: return TypeU32(state);
		case IR::Type::U64: return TypeU64(state);
		case IR::Type::U32x2: return TypeU32Pair(state);
		case IR::Type::F32: return TypeF32(state);
		case IR::Type::F64: return TypeF64(state);
		case IR::Type::U32x3: return TypeU32Vector(state, 3);
		case IR::Type::U32x4: return TypeU32Vector(state, 4);
		case IR::Type::F32x2: return TypeF32Vector(state, 2);
		default: return 0;
	}
}

uint32_t ValueEmitContext::Def(IR::Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case IR::Type::U1: return ConstantBool(state, value.U1());
			case IR::Type::U8: return ConstantU32(state, value.U8());
			case IR::Type::U16: return ConstantU32(state, value.U16());
			case IR::Type::U32: return ConstantU32(state, value.U32());
			case IR::Type::U64: return ConstantU64(state, value.U64());
			case IR::Type::F16: return ConstantU32(state, value.F16Bits());
			case IR::Type::F32:
				return ConstantF32(state, std::bit_cast<uint32_t>(value.F32Value()));
			default: break;
		}
	}
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr) {
		Fail("direct SPIR-V emitter received a non-value argument");
	}
	if (dispatcher_spills != nullptr && state.current_block != nullptr &&
	    inst->Parent() != state.current_block) {
		if (const auto found = dispatcher_spills->find(inst); found != dispatcher_spills->end()) {
			if (const auto loaded = dispatcher_block_loads.find(inst);
			    loaded != dispatcher_block_loads.end() &&
			    loaded->second.first == state.current_label) {
				return loaded->second.second;
			}
			const auto id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeId(state, inst->GetType()), id,
			                          found->second);
			dispatcher_block_loads.insert_or_assign(inst, std::pair {state.current_label, id});
			return id;
		}
	}
	return Result(*inst);
}

uint32_t ValueEmitContext::Arg(const IR::Inst& inst, size_t index) {
	return Def(inst.Arg(index));
}

uint32_t ValueEmitContext::HalfArg(const IR::Inst& inst, size_t index, uint32_t lane_half) {
	return lane_half == half ? Arg(inst, index) : other_half->Arg(inst, index);
}

uint32_t ValueEmitContext::Ballot(IR::Value predicate) {
	const auto ballot_type = TypeU32Vector(state, 4);
	const auto scope       = ConstantU32(state, spv::ScopeSubgroup);
	const auto low         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, low, scope,
	                          other_half == nullptr || half == 0 ? Def(predicate)
	                                                             : other_half->Def(predicate));
	if (other_half == nullptr) {
		return low;
	}
	const auto high      = state.builder.AllocateId();
	const auto low_word  = state.builder.AllocateId();
	const auto high_word = state.builder.AllocateId();
	const auto ballot    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, high, scope,
	                          half == 1 ? Def(predicate) : other_half->Def(predicate));
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low_word, low, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high_word, high, 0);
	state.builder.AddFunction(spv::OpCompositeConstruct, ballot_type, ballot, low_word, high_word,
	                          ConstantU32(state, 0), ConstantU32(state, 0));
	return ballot;
}

uint32_t ValueEmitContext::FirstLane(uint32_t ballot) {
	if (other_half == nullptr) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallotFindLSB, TypeU32(state), result,
		                          ConstantU32(state, spv::ScopeSubgroup), ballot);
		return result;
	}
	const auto low        = state.builder.AllocateId();
	const auto high       = state.builder.AllocateId();
	const auto low_first  = state.builder.AllocateId();
	const auto high_first = state.builder.AllocateId();
	const auto low_active = state.builder.AllocateId();
	const auto result     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), low_first, GlslStd450(state),
	                          GLSLstd450FindILsb, low);
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), high_first, GlslStd450(state),
	                          GLSLstd450FindILsb, high);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), low_active, low,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), result, low_active, low_first,
	                          EmitAddU32(state, high_first, ConstantU32(state, 32)));
	return result;
}

uint32_t ValueEmitContext::Shuffle(const IR::Inst& inst, size_t index, uint32_t lane) {
	const auto type  = TypeId(state, inst.Arg(index).GetType());
	const auto scope = ConstantU32(state, spv::ScopeSubgroup);
	const auto low   = state.builder.AllocateId();
	if (other_half == nullptr) {
		state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope, Arg(inst, index),
		                          lane);
		return low;
	}
	const auto physical_lane =
	    EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31));
	const auto high          = state.builder.AllocateId();
	const auto in_high       = state.builder.AllocateId();
	const auto value         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope,
	                          HalfArg(inst, index, 0), physical_lane);
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, high, scope,
	                          HalfArg(inst, index, 1), physical_lane);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), in_high,
	                          EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 32)),
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, type, value, in_high, high, low);
	return value;
}

uint32_t ValueEmitContext::Result(const IR::Inst& inst) {
	if (const auto id = inst.Definition(half); id != 0) {
		return id;
	}
	const auto id = state.builder.AllocateId();
	inst.SetDefinition(half, id);
	return id;
}

uint32_t ValueEmitContext::Define(const IR::Inst& inst, uint32_t value) {
	if (const auto id = inst.Definition(half); id != 0) {
		if (id != value) {
			state.builder.AddFunction(spv::OpCopyObject, TypeId(state, inst.GetType()),
			                          id, value);
		}
		return id;
	}
	inst.SetDefinition(half, value);
	return value;
}

uint32_t ValueEmitContext::ResourceIndex(IR::Value value, IR::ValueOpcode opcode) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != opcode) {
		Fail("typed resource handle has the wrong producer");
	}
	return inst->Flags<uint32_t>();
}

const IR::Inst* ValueEmitContext::ImageAddress(IR::Value value) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != IR::ValueOpcode::MakeImageAddress) {
		Fail("typed image address was not constructed by MakeImageAddress");
	}
	return inst;
}

const IR::MemoryInfo& ValueEmitContext::Memory(const IR::Inst& inst) const {
	return state.program.memory_info.at(inst.Flags<IR::MemoryFlags>().index);
}

const IR::ExportInfo& ValueEmitContext::Export(const IR::Inst& inst) const {
	return state.program.export_info.at(inst.Flags<IR::ExportFlags>().index);
}

[[noreturn]] void ValueEmitContext::Fail(const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage), reason);
	std::abort();
}

[[noreturn]] void ValueEmitContext::Fail(const IR::Inst& inst, const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u opcode=%s reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage),
	     IR::ValueOpcodeName(inst.GetOpcode()), reason);
	std::abort();
}

void EmitProgram(EmitterState& state) {
	const auto&      program = state.program;
	ValueEmitContext ctx(state);
	ValueEmitContext high(state);
	if (state.lane_count == 2) {
		ctx.other_half  = &high;
		high.other_half = &ctx;
		high.half       = 1;
	}
	std::optional<DispatcherFunctionState> dispatcher;
	if (state.program.stage == ShaderType::Pixel && state.program.info.pixel_valid_mask) {
		state.pixel_valid_mask_variable = state.builder.AllocateId();
		state.builder.AddName(state.pixel_valid_mask_variable, "pixel_valid_mask_active");
	}
	for (const auto* block: program.blocks) {
		// Re-emission may use a different host subgroup width and ID allocation order.
		for (const auto& inst: *block) {
			inst.ResetDefinitions();
		}
		block->SetDefinition(state.builder.AllocateId());
	}
	if (state.program.dispatcher_fallback) {
		auto& dispatch = dispatcher.emplace();
		// Allocate both halves in program order so the SPIR-V is identical across runs.
		const auto add_spill = [&](const IR::Inst* inst) {
			for (uint32_t half = 0; half < state.lane_count; half++) {
				dispatch.spills[half].emplace(inst, state.builder.AllocateId());
			}
		};
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				if (inst.GetOpcode() != IR::ValueOpcode::Phi) {
					continue;
				}
				if (SpillPointerType(ctx, inst.GetType()) == 0) {
					ctx.Fail(inst, "cannot be stored by the dispatcher");
					break;
				}
				add_spill(&inst);
			}
		}
		const auto mark_cross_block = [&](IR::Value value, const IR::Block* consumer) {
			value                  = value.Resolve();
			const auto* definition = value.TryInstruction();
			if (definition == nullptr || definition->Parent() == consumer ||
			    definition->Parent() == program.blocks.front()) {
				return;
			}
			if (SpillPointerType(ctx, definition->GetType()) == 0) {
				ctx.Fail(*definition, "cannot be stored by the dispatcher");
				return;
			}
			if (!dispatch.spills[0].contains(definition)) {
				add_spill(definition);
			}
		};
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				for (size_t index = 0; index < inst.NumArgs(); index++) {
					const auto* consumer =
					    inst.GetOpcode() == IR::ValueOpcode::Phi ? inst.PhiBlock(index) : block;
					mark_cross_block(inst.Arg(index), consumer);
				}
			}
		}
		for (size_t index = 0; index < program.blocks.size(); index++) {
			mark_cross_block(program.blocks[index]->condition, program.blocks[index]);
			mark_cross_block(program.blocks[index]->indirect_target, program.blocks[index]);
		}
		dispatch.header_label       = state.builder.AllocateId();
		dispatch.select_label       = state.builder.AllocateId();
		dispatch.after_switch_label = state.builder.AllocateId();
		dispatch.continue_label     = state.builder.AllocateId();
		dispatch.merge_label        = state.builder.AllocateId();
		ctx.dispatcher_spills       = &dispatch.spills[0];
		high.dispatcher_spills      = &dispatch.spills[1];
	}
	DefineGetBdaPointer(state);
	DefineIndirectBufferLoads(ctx);
	DefineIndirectBufferFormats(ctx);
	DefineBvhIntersect(state);
	if (program.info.uses_swizzle) {
		ctx.scratch_u32_variable = state.builder.AllocateId();
		if (state.lane_count == 2) {
			high.scratch_u32_variable = state.builder.AllocateId();
		}
	}
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state),
	                          state.mesh_guest_func != 0 ? state.mesh_guest_func : state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.entry_label);
	if (state.program.info.function_lds) {
		state.builder.AddFunction(
		    spv::OpVariable,
		    TypeU32ArrayPointer(state, spv::StorageClassFunction, LdsDwordCount(state)),
		    state.lds_variable, spv::StorageClassFunction);
	}
	if (state.program.info.function_scratch) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.builder.AddFunction(
			    spv::OpVariable,
			    TypeU32ArrayPointer(state, spv::StorageClassFunction, state.program.scratch_dwords),
			    state.scratch_variable[half], spv::StorageClassFunction);
		}
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.pixel_valid_mask_variable, spv::StorageClassFunction);
	}
	for (uint32_t half = 0; half < state.lane_count; half++) {
		auto& lane = half == 0 ? ctx : high;
		if (state.program.dispatcher_fallback) {
			for (const auto* block: program.blocks) {
				for (const auto& inst: *block) {
					if (const auto found = dispatcher->spills[half].find(&inst);
					    found != dispatcher->spills[half].end()) {
						state.builder.AddFunction(spv::OpVariable,
						                          SpillPointerType(lane, inst.GetType()),
						                          found->second, spv::StorageClassFunction);
					}
				}
			}
		}
		if (lane.scratch_u32_variable != 0) {
			state.builder.AddFunction(spv::OpVariable,
			                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
			                          lane.scratch_u32_variable, spv::StorageClassFunction);
		}
	}
	state.loop_watchdogs.clear();
	state.loop_watchdog_breaks.clear();
	state.loop_watchdog_waterfalls.clear();
	state.loop_watchdog_resets.clear();
	g_last_waterfall_guards = 0;
	static const bool watchdog = std::getenv("KYTY_NO_LOOP_WATCHDOG") == nullptr;
	if (watchdog && !state.program.dispatcher_fallback) {
		std::unordered_map<const IR::Block*, const IR::Block*> loop_merges;
		std::unordered_set<const IR::Block*>                   merge_blocks;
		for (const auto* block: program.blocks) {
			if (block->terminator.loop_header && block->terminator.merge_block != nullptr) {
				loop_merges.emplace(block, block->terminator.merge_block);
				merge_blocks.insert(block->terminator.merge_block);
			}
		}
		const auto exits_loop = [&](const IR::Block* block) {
			const auto& term = block->terminator;
			if (term.kind != CFG::TerminatorKind::ConditionalBranch || block->condition.IsEmpty() ||
			    term.true_block == nullptr || term.false_block == nullptr) {
				return false;
			}
			if (term.loop_header) {
				return term.merge_block != nullptr && term.continue_block != nullptr &&
				       (term.merge_block == term.true_block || term.merge_block == term.false_block);
			}
			for (const auto [header, other]: {std::pair {term.true_block, term.false_block},
			                                  std::pair {term.false_block, term.true_block}}) {
				const auto found = loop_merges.find(header);
				if (found != loop_merges.end() && found->second == other) return true;
			}
			return false;
		};
		// Waterfall loops: a ReadFirstLane of x compared against x picks one lane value per pass.
		// Loops span the blocks from their header up to their merge in program order.
		std::unordered_map<const IR::Block*, size_t> block_index;
		for (size_t i = 0; i < program.blocks.size(); i++) block_index.emplace(program.blocks[i], i);
		const auto is_waterfall_body = [&](size_t first, size_t last) {
			for (size_t i = first; i < last; i++) {
				for (const auto& inst: program.blocks[i]->Instructions()) {
					if (inst.GetOpcode() != IR::ValueOpcode::ReadFirstLane || inst.NumArgs() == 0) continue;
					const auto source = inst.Arg(0);
					for (size_t j = first; j < last; j++) {
						for (const auto& user: program.blocks[j]->Instructions()) {
							if (user.NumArgs() < 2) continue;
							bool uses_first = false, uses_source = false;
							for (size_t a = 0; a < user.NumArgs(); a++) {
								const auto arg = user.Arg(a);
								uses_first |= arg.TryInstruction() == &inst;
								uses_source |= arg == source;
							}
							if (uses_first && uses_source) return true;
						}
					}
				}
			}
			return false;
		};
		std::vector<std::pair<size_t, size_t>> waterfall_ranges;
		for (const auto& [header, merge]: loop_merges) {
			const auto first = block_index.find(header);
			const auto last  = block_index.find(merge);
			if (first != block_index.end() && last != block_index.end() && first->second < last->second &&
			    is_waterfall_body(first->second, last->second)) {
				waterfall_ranges.emplace_back(first->second, last->second);
			}
		}
		const auto in_waterfall = [&](const IR::Block* block) {
			const auto index = block_index.at(block);
			// The innermost loop around the block decides; an inner ordinary loop keeps its cap.
			size_t best_size = SIZE_MAX;
			bool   waterfall  = false;
			for (const auto& [header, merge]: loop_merges) {
				const auto first = block_index.find(header);
				const auto last  = block_index.find(merge);
				if (first == block_index.end() || last == block_index.end() || index < first->second ||
				    index >= last->second || last->second - first->second >= best_size) {
					continue;
				}
				best_size  = last->second - first->second;
				waterfall  = std::ranges::find(waterfall_ranges, std::pair {first->second, last->second}) !=
				            waterfall_ranges.end();
			}
			return waterfall;
		};
		// The merge block of the innermost loop around a block (counters restart there).
		const auto innermost_merge = [&](const IR::Block* block) -> const IR::Block* {
			const auto index     = block_index.at(block);
			size_t     best_size = SIZE_MAX;
			const IR::Block* best = nullptr;
			for (const auto& [header, merge]: loop_merges) {
				const auto first = block_index.find(header);
				const auto last  = block_index.find(merge);
				if (first == block_index.end() || last == block_index.end() || index < first->second ||
				    index >= last->second || last->second - first->second >= best_size) {
					continue;
				}
				best_size = last->second - first->second;
				best      = merge;
			}
			return best;
		};
		// KYTY_LOOP_COUNTER_NO_RESET: old behaviour, one count per invocation for all entries of a
		// loop (a loop inside another loop hit its cap early and silently skipped work).
		static const bool reset_counters = std::getenv("KYTY_LOOP_COUNTER_NO_RESET") == nullptr;
		for (const auto* block: program.blocks) {
			const auto& term = block->terminator;
			bool guarded = exits_loop(block);
			if (!guarded && !term.loop_header &&
			    term.kind == CFG::TerminatorKind::ConditionalBranch && !block->condition.IsEmpty() &&
			    term.true_block != nullptr && term.false_block != nullptr &&
			    merge_blocks.contains(term.true_block) != merge_blocks.contains(term.false_block)) {
				// A break out of a loop body: exits that only lanes running out of work take.
				state.loop_watchdog_breaks.emplace(block, merge_blocks.contains(term.true_block));
				guarded = true;
			}
			if (guarded) {
				const auto counter = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpVariable,
				                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
				                          counter, spv::StorageClassFunction);
				state.loop_watchdogs.emplace(block, counter);
				if (!waterfall_ranges.empty() && in_waterfall(block)) {
					state.loop_watchdog_waterfalls.insert(block);
					g_last_waterfall_guards++;
					// Waterfall loops end on their own (each pass retires a lane): restart their count
					// each time they are entered. Other loops keep one count per invocation, which
					// bounds runaway nested loops (resetting those too gave 1.3 s GPU stalls in GT7).
					if (reset_counters) {
						if (const auto* merge = innermost_merge(block); merge != nullptr) {
							state.loop_watchdog_resets[merge].push_back(counter);
						}
					}
				}
			}
		}
	}
	if (state.gds_variable != 0) {
		state.gds_length = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), state.gds_length,
		                          state.gds_variable, 0);
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable,
		                          ConstantU32(state, 1));
	}
	if (state.lds_storage_class == spv::StorageClassStorageBuffer && state.lds_variable != 0) {
		const auto group_x = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 0);
		const auto group_y = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 1);
		const auto group_z = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 2);
		const auto count_x = EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, 0);
		const auto count_y = EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, 1);
		const auto row = EmitAddU32(state, group_y, EmitBinaryU32(state, spv::OpIMul, group_z, count_y));
		const auto index = EmitAddU32(state, group_x, EmitBinaryU32(state, spv::OpIMul, row, count_x));
		state.lds_base_dwords = EmitBinaryU32(state, spv::OpIMul, index,
		                                     ConstantU32(state, LdsDwordCount(state)));
	}
	// Program order, not map order: the map follows addresses, which change between runs.
	for (const auto* block: program.blocks) {
		if (const auto guard = state.loop_watchdogs.find(block); guard != state.loop_watchdogs.end()) {
			state.builder.AddFunction(spv::OpStore, guard->second, ConstantU32(state, 0));
		}
	}
	EmitMemoryOffsets(state);
	if (program.blocks.empty()) {
		EmitReturn(ctx);
	} else if (state.program.dispatcher_fallback) {
		EmitDispatcherFunction(ctx, *dispatcher);
	} else {
		EmitStructuredFunction(ctx);
	}
	state.builder.AddFunction(spv::OpFunctionEnd);
	state.current_block = nullptr;
	if (state.program.stage == ShaderType::Mesh) {
		EmitMeshEntryPoint(state);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
