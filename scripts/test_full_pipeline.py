#!/usr/bin/env python3
"""
Full Pipeline Test for STAIR-RL

Orchestrates end-to-end testing of the STAIR-RL training pipeline:
1. Generate mock data from OHLCV
2. Preprocess to tensor format
3. Run Phase 1 (CQL-SAC) training
4. Run Phase 2 (PPO-CVaR) training
5. Generate validation report

Usage:
    # Run full pipeline
    python scripts/test_full_pipeline.py --run-all

    # Or run individual steps
    python scripts/test_full_pipeline.py --step generate
    python scripts/test_full_pipeline.py --step preprocess
    python scripts/test_full_pipeline.py --step phase1
    python scripts/test_full_pipeline.py --step phase2
    python scripts/test_full_pipeline.py --step validate
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineTester:
    """
    Orchestrates full pipeline testing.
    """

    def __init__(
        self,
        output_dir: Path = Path('/home/work/data/stair-local/test_mock'),
        checkpoint_dir: Path = Path('/tmp/stair_test_checkpoints'),
        n_assets: int = 10,
        start_date: str = '2021-01-01',
        end_date: str = '2021-02-28',
    ):
        """
        Initialize pipeline tester.

        Args:
            output_dir: Output directory for mock data
            checkpoint_dir: Directory for checkpoints
            n_assets: Number of assets to test
            start_date: Start date
            end_date: End date
        """
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.n_assets = n_assets
        self.start_date = start_date
        self.end_date = end_date

        self.scripts_dir = Path(__file__).parent
        self.tensor_dir = output_dir / 'tensors'

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Test results
        self.results: Dict[str, bool] = {}
        self.timings: Dict[str, float] = {}

        logger.info("Initialized Pipeline Tester")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Checkpoints: {checkpoint_dir}")
        logger.info(f"  Assets: {n_assets}")
        logger.info(f"  Period: {start_date} to {end_date}")

    def _run_command(self, cmd: List[str], step_name: str) -> bool:
        """
        Run a command and track timing.

        Args:
            cmd: Command list
            step_name: Step name for logging

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running: {step_name}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info(f"{'=' * 60}\n")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True,
            )

            elapsed = time.time() - start_time
            self.timings[step_name] = elapsed

            logger.info(f"\n✓ {step_name} completed in {elapsed:.1f}s\n")
            return True

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            self.timings[step_name] = elapsed

            logger.error(f"\n✗ {step_name} failed after {elapsed:.1f}s")
            logger.error(f"Error: {e}")
            return False

    def step_generate(self) -> bool:
        """Step 1: Generate mock data."""
        cmd = [
            'python', str(self.scripts_dir / 'generate_mock_data.py'),
            '--n-assets', str(self.n_assets),
            '--start-date', self.start_date,
            '--end-date', self.end_date,
            '--output', str(self.output_dir),
        ]

        success = self._run_command(cmd, 'Generate Mock Data')
        self.results['generate'] = success
        return success

    def step_preprocess(self) -> bool:
        """Step 2: Preprocess to tensors."""
        cmd = [
            'python', str(self.scripts_dir / 'prepare_mock_training_data.py'),
            '--input', str(self.output_dir),
            '--output', str(self.tensor_dir),
        ]

        success = self._run_command(cmd, 'Preprocess Data')
        self.results['preprocess'] = success
        return success

    def step_phase1(self) -> bool:
        """Step 3: Phase 1 training."""
        train_data = self.tensor_dir / 'train_data.npz'

        if not train_data.exists():
            logger.error(f"Training data not found: {train_data}")
            self.results['phase1'] = False
            return False

        phase1_checkpoint_dir = self.checkpoint_dir / 'phase1'
        phase1_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'python', str(self.scripts_dir / 'test_training_minimal.py'),
            '--phase', '1',
            '--data', str(train_data),
            '--checkpoint-dir', str(phase1_checkpoint_dir),
            '--steps', '1000',
            '--n-assets', str(self.n_assets),
            '--state-dim', '36',
            '--device', 'cpu',
        ]

        success = self._run_command(cmd, 'Phase 1 Training (CQL-SAC)')
        self.results['phase1'] = success
        return success

    def step_phase2(self) -> bool:
        """Step 4: Phase 2 training."""
        val_data = self.tensor_dir / 'val_data.npz'
        pretrained = self.checkpoint_dir / 'phase1' / 'cql_sac_final.pt'

        if not val_data.exists():
            logger.error(f"Validation data not found: {val_data}")
            self.results['phase2'] = False
            return False

        phase2_checkpoint_dir = self.checkpoint_dir / 'phase2'
        phase2_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'python', str(self.scripts_dir / 'test_training_minimal.py'),
            '--phase', '2',
            '--data', str(val_data),
            '--checkpoint-dir', str(phase2_checkpoint_dir),
            '--steps', '500',
            '--n-assets', str(self.n_assets),
            '--state-dim', '36',
            '--device', 'cpu',
        ]

        if pretrained.exists():
            cmd.extend(['--pretrained', str(pretrained)])

        success = self._run_command(cmd, 'Phase 2 Training (PPO-CVaR)')
        self.results['phase2'] = success
        return success

    def validate(self) -> bool:
        """Step 5: Validate all components."""
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION")
        logger.info("=" * 60)

        all_passed = True

        # Check data generation
        features_dir = self.output_dir / 'features' / 'mock_features'
        if features_dir.exists():
            feature_files = list(features_dir.glob('*_features.parquet'))
            data_check = len(feature_files) == self.n_assets
            logger.info(f"✓ Data Generation: {len(feature_files)}/{self.n_assets} feature files")
        else:
            data_check = False
            logger.error("✗ Data Generation: Features directory not found")
        all_passed &= data_check

        # Check preprocessing
        train_data = self.tensor_dir / 'train_data.npz'
        val_data = self.tensor_dir / 'val_data.npz'
        preprocess_check = train_data.exists() and val_data.exists()
        if preprocess_check:
            logger.info(f"✓ Preprocessing: Train and val tensors created")
        else:
            logger.error("✗ Preprocessing: Tensor files missing")
        all_passed &= preprocess_check

        # Check Phase 1
        phase1_checkpoint = self.checkpoint_dir / 'phase1' / 'cql_sac_final.pt'
        phase1_check = phase1_checkpoint.exists()
        if phase1_check:
            logger.info(f"✓ Phase 1: Checkpoint saved")
        else:
            logger.error("✗ Phase 1: Checkpoint missing")
        all_passed &= phase1_check

        # Check Phase 2
        phase2_checkpoint = self.checkpoint_dir / 'phase2' / 'ppo_cvar_final.pt'
        phase2_check = phase2_checkpoint.exists()
        if phase2_check:
            logger.info(f"✓ Phase 2: Checkpoint saved")
        else:
            logger.error("✗ Phase 2: Checkpoint missing")
        all_passed &= phase2_check

        self.results['validate'] = all_passed
        return all_passed

    def generate_report(self):
        """Generate final test report."""
        logger.info("\n\n" + "=" * 60)
        logger.info("STAIR-RL TRAINING PIPELINE TEST REPORT")
        logger.info("=" * 60)

        # Results
        logger.info("\nTest Results:")
        for step, passed in self.results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {step.ljust(20)}: {status}")

        # Timings
        logger.info("\nExecution Time:")
        total_time = sum(self.timings.values())
        for step, elapsed in self.timings.items():
            logger.info(f"  {step.ljust(20)}: {elapsed:.1f}s")
        logger.info(f"  {'Total'.ljust(20)}: {total_time:.1f}s")

        # Overall status
        all_passed = all(self.results.values())
        logger.info("\n" + "=" * 60)
        if all_passed:
            logger.info("✓ ALL TESTS PASSED")
            logger.info("Training pipeline is ready for full-scale training.")
        else:
            logger.error("✗ SOME TESTS FAILED")
            logger.error("Please review the errors above.")
        logger.info("=" * 60 + "\n")

        return all_passed

    def run_all(self) -> bool:
        """Run all steps in sequence."""
        steps = [
            ('generate', self.step_generate),
            ('preprocess', self.step_preprocess),
            ('phase1', self.step_phase1),
            ('phase2', self.step_phase2),
            ('validate', self.validate),
        ]

        for step_name, step_func in steps:
            success = step_func()
            if not success:
                logger.error(f"Pipeline stopped at step: {step_name}")
                break

        return self.generate_report()


def main():
    parser = argparse.ArgumentParser(description='Test full training pipeline')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all steps')
    parser.add_argument('--step', type=str, choices=[
                        'generate', 'preprocess', 'phase1', 'phase2', 'validate'],
                        help='Run specific step')
    parser.add_argument('--output', type=str,
                        default='/home/work/data/stair-local/test_mock',
                        help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/tmp/stair_test_checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--n-assets', type=int, default=10,
                        help='Number of assets')
    parser.add_argument('--start-date', type=str, default='2021-01-01',
                        help='Start date')
    parser.add_argument('--end-date', type=str, default='2021-02-28',
                        help='End date')

    args = parser.parse_args()

    if not args.run_all and not args.step:
        parser.error("Must specify either --run-all or --step")

    tester = PipelineTester(
        output_dir=Path(args.output),
        checkpoint_dir=Path(args.checkpoint_dir),
        n_assets=args.n_assets,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if args.run_all:
        success = tester.run_all()
    else:
        step_map = {
            'generate': tester.step_generate,
            'preprocess': tester.step_preprocess,
            'phase1': tester.step_phase1,
            'phase2': tester.step_phase2,
            'validate': tester.validate,
        }
        success = step_map[args.step]()
        tester.generate_report()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
