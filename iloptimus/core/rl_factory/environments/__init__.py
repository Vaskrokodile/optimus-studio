"""Environments package for rl-factory."""

from iloptimus.core.rl_factory.environments.adversarial_code_transform import AdversarialCodeTransformEnv, adversarial_code_transform_generator
from iloptimus.core.rl_factory.environments.adversarial_counterfactual import AdversarialCounterfactualEnv, adversarial_counterfactual_generator
from iloptimus.core.rl_factory.environments.algorithm_invention import AlgorithmInventionEnv, algorithm_invention_generator
from iloptimus.core.rl_factory.environments.answer_first import AnswerFirstEnv, answer_first_generator
from iloptimus.core.rl_factory.environments.art_painting_java import ArtPaintingJavaEnv, art_painting_generator
from iloptimus.core.rl_factory.environments.ast_refactor_chain import AstRefactorChainEnv, ast_refactor_chain_generator
from iloptimus.core.rl_factory.environments.best_of_n import BestOfNEnv, bestofn_generator
from iloptimus.core.rl_factory.environments.binary_analysis import BinaryAnalysisEnv, binary_analysis_generator
from iloptimus.core.rl_factory.environments.breakthrough_mechanism_mapping import BreakthroughMechanismMappingEnv, breakthrough_mechanism_mapping_generator
from iloptimus.core.rl_factory.environments.bug_hunt_checkpoints import BugHuntCheckpointsEnv, bug_hunt_checkpoints_generator
from iloptimus.core.rl_factory.environments.calibrated_qa import CalibratedQAEnv, calibrated_qa_generator
from iloptimus.core.rl_factory.environments.causal_surgery import CausalSurgeryEnv, causal_surgery_generator
from iloptimus.core.rl_factory.environments.checkpoint_auditor import CheckpointAuditorEnv, checkpoint_auditor_generator
from iloptimus.core.rl_factory.environments.checkpoint_executor import CheckpointExecutorEnv, checkpoint_executor_generator
from iloptimus.core.rl_factory.environments.checkpoint_planner import CheckpointPlannerEnv, checkpoint_planner_generator
from iloptimus.core.rl_factory.environments.checkpoint_race import CheckpointRaceEnv, checkpoint_race_generator
from iloptimus.core.rl_factory.environments.checkpoint_recovery import CheckpointRecoveryEnv, checkpoint_recovery_generator
from iloptimus.core.rl_factory.environments.code_review_checkpoints import CodeReviewCheckpointsEnv, code_review_checkpoints_generator
from iloptimus.core.rl_factory.environments.compiler_opt_arena import CompilerOptArenaEnv, compiler_opt_arena_generator
from iloptimus.core.rl_factory.environments.constraint_formalization import ConstraintFormalizationEnv, constraint_formalization_generator
from iloptimus.core.rl_factory.environments.context_distiller import ContextDistillerEnv, context_distiller_generator
from iloptimus.core.rl_factory.environments.context_hunter import ContextHunterEnv, context_hunter_generator
from iloptimus.core.rl_factory.environments.contradiction_detection import ContradictionDetectionEnv, contradiction_detection_generator
from iloptimus.core.rl_factory.environments.corpus_grounded_self_play import CorpusGroundedSelfPlayEnv, corpus_grounded_self_play_generator
from iloptimus.core.rl_factory.environments.cross_domain_transfer import CrossDomainTransferEnv, cross_domain_transfer_generator
from iloptimus.core.rl_factory.environments.crypto_quest import CryptoQuestEnv, crypto_quest_generator
from iloptimus.core.rl_factory.environments.debugging_race import DebuggingRaceEnv, debugging_race_generator
from iloptimus.core.rl_factory.environments.dependency_chain_builder import DependencyChainBuilderEnv, dependency_chain_builder_generator
from iloptimus.core.rl_factory.environments.diff_judge import DiffJudgeEnv, diff_judge_generator
from iloptimus.core.rl_factory.environments.diff_level_repair import DiffLevelRepairEnv, diff_level_repair_generator
from iloptimus.core.rl_factory.environments.environment_mutation import EnvironmentMutationEnv, environment_mutation_generator
from iloptimus.core.rl_factory.environments.error_recovery import ErrorRecoveryEnv, error_recovery_generator
from iloptimus.core.rl_factory.environments.experiment_design_from_hypothesis import ExperimentDesignFromHypothesisEnv, experiment_design_from_hypothesis_generator
from iloptimus.core.rl_factory.environments.failure_first import FailureFirstEnv, failure_first_generator
from iloptimus.core.rl_factory.environments.formal_proof_golf_env import FormalProofGolfEnv, formal_proof_golf_generator
from iloptimus.core.rl_factory.environments.formal_proof_gym import FormalProofGymEnv, formal_proof_gym_generator
from iloptimus.core.rl_factory.environments.hybrid_verification import HybridVerificationEnv, hybrid_verification_generator
from iloptimus.core.rl_factory.environments.implicit_process_reward import ImplicitProcessRewardEnv, implicit_process_reward_generator
from iloptimus.core.rl_factory.environments.incremental_complexity import IncrementalComplexityEnv, incremental_complexity_generator
from iloptimus.core.rl_factory.environments.info_asymmetry_tasks import InfoAsymmetryEnv, info_asymmetry_tasks_generator
from iloptimus.core.rl_factory.environments.instruction_decomposition import InstructionDecompositionEnv, instruction_decomposition_generator
from iloptimus.core.rl_factory.environments.iterative_refinement import IterativeRefinementEnv, iterative_refinement_generator
from iloptimus.core.rl_factory.environments.kernel_build_checkpoints import KernelBuildCheckpointsEnv, kernel_build_checkpoints_generator
from iloptimus.core.rl_factory.environments.kernel_optimizer import KernelOptimizerEnv, kernel_optimizer_generator
from iloptimus.core.rl_factory.environments.logic_puzzle_suite import LogicPuzzleSuiteEnv, logic_puzzle_suite_generator
from iloptimus.core.rl_factory.environments.math_gap_arithmetic import MathGapArithmeticEnv, math_gap_arithmetic_generator
from iloptimus.core.rl_factory.environments.meta_pattern_extraction import MetaPatternExtractionEnv, meta_pattern_extraction_generator
from iloptimus.core.rl_factory.environments.minimum_information import MinimumInformationEnv, minimum_information_generator
from iloptimus.core.rl_factory.environments.novel_hypothesis_from_evidence import NovelHypothesisFromEvidenceEnv, novel_hypothesis_from_evidence_generator
from iloptimus.core.rl_factory.environments.multi_objective_curriculum import MultiObjectiveCurriculumEnv, multi_objective_curriculum_generator
from iloptimus.core.rl_factory.environments.obfuscation_arena import ObfuscationArenaEnv, obfuscation_arena_generator
from iloptimus.core.rl_factory.environments.parallel_explorer import ParallelExplorerEnv, parallel_explorer_generator
from iloptimus.core.rl_factory.environments.patch_guard import PatchGuardEnv, patch_guard_generator
from iloptimus.core.rl_factory.environments.performance_profiling import PerformanceProfilingEnv, performance_profiling_generator
from iloptimus.core.rl_factory.environments.plan_adherence import PlanAdherenceEnv, plan_adherence_generator
from iloptimus.core.rl_factory.environments.prompt_guard import PromptGuardEnv, prompt_guard_generator
from iloptimus.core.rl_factory.environments.proof_compression import ProofCompressionEnv, proof_compression_generator
from iloptimus.core.rl_factory.environments.proof_golf import ProofGolfEnv, proofgolf_generator
from iloptimus.core.rl_factory.environments.property_based_contracts import PropertyBasedContractsEnv, property_based_contracts_generator
from iloptimus.core.rl_factory.environments.pursuit_evasion_curriculum import PursuitEvasionCurriculumEnv, pursuit_evasion_curriculum_generator
from iloptimus.core.rl_factory.environments.redundancy_eliminator import RedundancyEliminatorEnv, redundancy_eliminator_generator
from iloptimus.core.rl_factory.environments.refactor_arena import RefactorArenaEnv, refactor_arena_generator
from iloptimus.core.rl_factory.environments.research_critique_and_gap import ResearchCritiqueAndGapEnv, research_critique_and_gap_generator
from iloptimus.core.rl_factory.environments.research_methodology_reconstruction import ResearchMethodologyReconstructionEnv, research_methodology_reconstruction_generator
from iloptimus.core.rl_factory.environments.refactor_checkpoint_sequence import RefactorCheckpointSequenceEnv, refactor_checkpoint_sequence_generator
from iloptimus.core.rl_factory.environments.reverse_curriculum import ReverseCurriculumEnv, reverse_curriculum_generator
from iloptimus.core.rl_factory.environments.sat_heuristic_discovery import SatHeuristicDiscoveryEnv, sat_heuristic_discovery_generator
from iloptimus.core.rl_factory.environments.self_consistency import SelfConsistencyEnv, self_consistency_generator
from iloptimus.core.rl_factory.environments.self_debug import SelfDebugEnv, self_debug_generator
from iloptimus.core.rl_factory.environments.shell_quest import ShellQuestEnv, shell_quest_generator
from iloptimus.core.rl_factory.environments.skill_collision import SkillCollisionEnv, skill_collision_generator
from iloptimus.core.rl_factory.environments.spec_to_verified_code import SpecToVerifiedCodeEnv, spec_to_verified_code_generator
from iloptimus.core.rl_factory.environments.specification_from_examples import SpecificationFromExamplesEnv, specification_from_examples_generator
from iloptimus.core.rl_factory.environments.speed_run import SpeedRunEnv, speed_run_generator
from iloptimus.core.rl_factory.environments.surgical_patch import SurgicalPatchEnv, surgical_patch_generator
from iloptimus.core.rl_factory.environments.symbolic_equation_solver import SymbolicEquationSolverEnv, symbolic_equation_solver_generator
from iloptimus.core.rl_factory.environments.test_driven_checkpoints import TestDrivenCheckpointsEnv, test_driven_checkpoints_generator
from iloptimus.core.rl_factory.environments.token_budget_golf import TokenBudgetGolfEnv, token_budget_golf_generator
from iloptimus.core.rl_factory.environments.tool_golf import ToolGolfEnv, toolgolf_generator
from iloptimus.core.rl_factory.environments.tournament_judge import TournamentJudgeEnv, tournament_judge_generator
from iloptimus.core.rl_factory.environments.trace_surgery import TraceSurgeryEnv, trace_surgery_generator
from iloptimus.core.rl_factory.environments.trajectory_doctor import TrajectoryDoctorEnv, trajectory_doctor_generator
from iloptimus.core.rl_factory.environments.trajectory_recombination import TrajectoryRecombinationEnv, trajectory_recombination_generator

__all__ = [
    "AdversarialCodeTransformEnv", "adversarial_code_transform_generator",
    "AdversarialCounterfactualEnv", "adversarial_counterfactual_generator",
    "AlgorithmInventionEnv", "algorithm_invention_generator",
    "AnswerFirstEnv", "answer_first_generator",
    "ArtPaintingJavaEnv", "art_painting_generator",
    "AstRefactorChainEnv", "ast_refactor_chain_generator",
    "BestOfNEnv", "bestofn_generator",
    "BinaryAnalysisEnv", "binary_analysis_generator",
    "BreakthroughMechanismMappingEnv", "breakthrough_mechanism_mapping_generator",
    "BugHuntCheckpointsEnv", "bug_hunt_checkpoints_generator",
    "CalibratedQAEnv", "calibrated_qa_generator",
    "CausalSurgeryEnv", "causal_surgery_generator",
    "CheckpointAuditorEnv", "checkpoint_auditor_generator",
    "CheckpointExecutorEnv", "checkpoint_executor_generator",
    "CheckpointPlannerEnv", "checkpoint_planner_generator",
    "CheckpointRaceEnv", "checkpoint_race_generator",
    "CheckpointRecoveryEnv", "checkpoint_recovery_generator",
    "CodeReviewCheckpointsEnv", "code_review_checkpoints_generator",
    "CompilerOptArenaEnv", "compiler_opt_arena_generator",
    "ConstraintFormalizationEnv", "constraint_formalization_generator",
    "ContextDistillerEnv", "context_distiller_generator",
    "ContextHunterEnv", "context_hunter_generator",
    "ContradictionDetectionEnv", "contradiction_detection_generator",
    "CorpusGroundedSelfPlayEnv", "corpus_grounded_self_play_generator",
    "CrossDomainTransferEnv", "cross_domain_transfer_generator",
    "CryptoQuestEnv", "crypto_quest_generator",
    "DebuggingRaceEnv", "debugging_race_generator",
    "DependencyChainBuilderEnv", "dependency_chain_builder_generator",
    "DiffJudgeEnv", "diff_judge_generator",
    "DiffLevelRepairEnv", "diff_level_repair_generator",
    "EnvironmentMutationEnv", "environment_mutation_generator",
    "ErrorRecoveryEnv", "error_recovery_generator",
    "ExperimentDesignFromHypothesisEnv", "experiment_design_from_hypothesis_generator",
    "FailureFirstEnv", "failure_first_generator",
    "FormalProofGolfEnv", "formal_proof_golf_generator",
    "FormalProofGymEnv", "formal_proof_gym_generator",
    "HybridVerificationEnv", "hybrid_verification_generator",
    "ImplicitProcessRewardEnv", "implicit_process_reward_generator",
    "IncrementalComplexityEnv", "incremental_complexity_generator",
    "InfoAsymmetryEnv", "info_asymmetry_tasks_generator",
    "InstructionDecompositionEnv", "instruction_decomposition_generator",
    "IterativeRefinementEnv", "iterative_refinement_generator",
    "KernelBuildCheckpointsEnv", "kernel_build_checkpoints_generator",
    "KernelOptimizerEnv", "kernel_optimizer_generator",
    "LogicPuzzleSuiteEnv", "logic_puzzle_suite_generator",
    "MathGapArithmeticEnv", "math_gap_arithmetic_generator",
    "MetaPatternExtractionEnv", "meta_pattern_extraction_generator",
    "MinimumInformationEnv", "minimum_information_generator",
    "NovelHypothesisFromEvidenceEnv", "novel_hypothesis_from_evidence_generator",
    "MultiObjectiveCurriculumEnv", "multi_objective_curriculum_generator",
    "ObfuscationArenaEnv", "obfuscation_arena_generator",
    "ParallelExplorerEnv", "parallel_explorer_generator",
    "PatchGuardEnv", "patch_guard_generator",
    "PerformanceProfilingEnv", "performance_profiling_generator",
    "PlanAdherenceEnv", "plan_adherence_generator",
    "PromptGuardEnv", "prompt_guard_generator",
    "ProofCompressionEnv", "proof_compression_generator",
    "ProofGolfEnv", "proofgolf_generator",
    "PropertyBasedContractsEnv", "property_based_contracts_generator",
    "PursuitEvasionCurriculumEnv", "pursuit_evasion_curriculum_generator",
    "RedundancyEliminatorEnv", "redundancy_eliminator_generator",
    "RefactorArenaEnv", "refactor_arena_generator",
    "ResearchCritiqueAndGapEnv", "research_critique_and_gap_generator",
    "ResearchMethodologyReconstructionEnv", "research_methodology_reconstruction_generator",
    "RefactorCheckpointSequenceEnv", "refactor_checkpoint_sequence_generator",
    "ReverseCurriculumEnv", "reverse_curriculum_generator",
    "SatHeuristicDiscoveryEnv", "sat_heuristic_discovery_generator",
    "SelfConsistencyEnv", "self_consistency_generator",
    "SelfDebugEnv", "self_debug_generator",
    "ShellQuestEnv", "shell_quest_generator",
    "SkillCollisionEnv", "skill_collision_generator",
    "SpecToVerifiedCodeEnv", "spec_to_verified_code_generator",
    "SpecificationFromExamplesEnv", "specification_from_examples_generator",
    "SpeedRunEnv", "speed_run_generator",
    "SurgicalPatchEnv", "surgical_patch_generator",
    "SymbolicEquationSolverEnv", "symbolic_equation_solver_generator",
    "TestDrivenCheckpointsEnv", "test_driven_checkpoints_generator",
    "TokenBudgetGolfEnv", "token_budget_golf_generator",
    "ToolGolfEnv", "toolgolf_generator",
    "TournamentJudgeEnv", "tournament_judge_generator",
    "TraceSurgeryEnv", "trace_surgery_generator",
    "TrajectoryDoctorEnv", "trajectory_doctor_generator",
    "TrajectoryRecombinationEnv", "trajectory_recombination_generator",
]
