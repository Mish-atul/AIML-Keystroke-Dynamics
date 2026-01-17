#!/usr/bin/env python
"""
Smoke Tests
===========
Quick validation tests to verify the pipeline works correctly.

Run all tests:
    python scripts/run_smoke_tests.py

Individual tests:
    python scripts/run_smoke_tests.py --test data
    python scripts/run_smoke_tests.py --test encoder
    python scripts/run_smoke_tests.py --test adapter
    python scripts/run_smoke_tests.py --test verify
"""

import os
import sys
import argparse
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def test_data_loading():
    """Test data loading and preprocessing."""
    print("\n" + "="*60)
    print("TEST: Data Loading & Preprocessing")
    print("="*60)
    
    from contrastive_encoder.data.preprocessing import (
        prepare_sequences,
        copy_and_normalize_dataset,
    )
    
    # Check if dataset exists
    csv_path = "cmu_keystroke.csv"
    original_path = "model training/DSL-StrongPasswordData.csv"
    
    if not os.path.exists(csv_path):
        if os.path.exists(original_path):
            print(f"Copying dataset from {original_path}...")
            copy_and_normalize_dataset(
                original_path, csv_path, "data_manifest.json"
            )
        else:
            print(f"ERROR: Dataset not found at {csv_path} or {original_path}")
            return False
    
    # Load data
    print("Loading and preprocessing data...")
    data = prepare_sequences(csv_path)
    
    # Validate shapes
    X_seq = data["X_seq"]
    X_agg = data["X_agg"]
    y = data["y"]
    
    assert X_seq.shape[1] == 11, f"Expected seq_len=11, got {X_seq.shape[1]}"
    assert X_seq.shape[2] == 3, f"Expected features=3, got {X_seq.shape[2]}"
    assert X_agg.shape[1] == 47, f"Expected 47 features, got {X_agg.shape[1]}"
    assert len(y) == len(X_seq), "Label count mismatch"
    
    n_users = len(np.unique(y))
    print(f"✓ Loaded {len(X_seq)} samples from {n_users} users")
    print(f"✓ X_seq shape: {X_seq.shape}")
    print(f"✓ X_agg shape: {X_agg.shape}")
    
    # Test augmentations
    from contrastive_encoder.data.augmentations import get_contrastive_augmentations
    aug = get_contrastive_augmentations(seed=42)
    sample = X_seq[0]
    augmented = aug(sample)
    
    assert augmented.shape == sample.shape, "Augmentation changed shape"
    print("✓ Augmentations working")
    
    print("\n✓ Data loading test PASSED")
    return True


def test_encoder():
    """Test encoder architecture and forward pass."""
    print("\n" + "="*60)
    print("TEST: Encoder Architecture")
    print("="*60)
    
    from contrastive_encoder.models.encoder import (
        KeystrokeEncoder,
        create_encoder,
    )
    from contrastive_encoder.models.losses import NTXentLoss, SupConLoss
    
    # Create encoder
    encoder = KeystrokeEncoder()
    print(f"Encoder parameters: {encoder.count_parameters():,}")
    
    # Forward pass
    batch = torch.randn(32, 11, 3)
    output = encoder(batch, normalize=True)
    
    assert output.shape == (32, 128), f"Expected (32, 128), got {output.shape}"
    
    # Check normalization
    norms = torch.norm(output, dim=1)
    assert torch.allclose(norms, torch.ones(32), atol=1e-5), "Embeddings not normalized"
    print("✓ Encoder forward pass working")
    print("✓ Output shape: (32, 128)")
    print("✓ L2 normalization verified")
    
    # Test losses
    z1 = torch.randn(32, 128)
    z2 = torch.randn(32, 128)
    labels = torch.randint(0, 10, (32,))
    
    ntxent = NTXentLoss()
    loss_ntxent = ntxent(z1, z2)
    assert not torch.isnan(loss_ntxent), "NT-Xent loss is NaN"
    print(f"✓ NT-Xent loss: {loss_ntxent.item():.4f}")
    
    supcon = SupConLoss()
    loss_supcon = supcon(torch.cat([z1, z2]), torch.cat([labels, labels]))
    assert not torch.isnan(loss_supcon), "SupCon loss is NaN"
    print(f"✓ SupCon loss: {loss_supcon.item():.4f}")
    
    print("\n✓ Encoder test PASSED")
    return True


def test_adapter():
    """Test adapter architecture and training."""
    print("\n" + "="*60)
    print("TEST: Adapter Architecture")
    print("="*60)
    
    from contrastive_encoder.models.adapter import UserAdapter, create_adapter
    
    # Create adapter
    adapter = create_adapter(hidden_dim=64)
    params = adapter.count_parameters()
    print(f"Adapter parameters: {params:,}")
    
    assert params < 20000, f"Adapter too large: {params} > 20000"
    print(f"✓ Adapter size under limit: {params:,} < 20,000")
    
    # Forward pass
    embeddings = torch.randn(10, 128)
    output = adapter(embeddings, normalize=True)
    
    assert output.shape == (10, 128), f"Expected (10, 128), got {output.shape}"
    
    # Check normalization
    norms = torch.norm(output, dim=1)
    assert torch.allclose(norms, torch.ones(10), atol=1e-5), "Output not normalized"
    print("✓ Adapter forward pass working")
    
    print("\n✓ Adapter test PASSED")
    return True


def test_quick_training():
    """Test quick encoder training (2 epochs)."""
    print("\n" + "="*60)
    print("TEST: Quick Training (2 epochs)")
    print("="*60)
    
    from contrastive_encoder.data.preprocessing import prepare_sequences
    from contrastive_encoder.training.train_encoder import EncoderTrainer
    from contrastive_encoder.config import EncoderConfig
    
    # Check dataset
    if not os.path.exists("cmu_keystroke.csv"):
        print("Skipping: dataset not available")
        return True
    
    # Load small subset
    data = prepare_sequences("cmu_keystroke.csv")
    
    # Take subset
    subset_size = 500
    indices = np.random.choice(len(data["X_seq_scaled"]), subset_size, replace=False)
    
    train_data = {
        "X_seq": data["X_seq_scaled"][indices],
        "y": data["y"][indices],
    }
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = EncoderConfig(
            epochs=2,
            batch_size=32,
            small_batch_mode=True,
        )
        
        trainer = EncoderTrainer(
            config=config,
            use_wandb=False,
            use_tensorboard=False,
            log_dir=os.path.join(temp_dir, "logs"),
        )
        
        encoder_path = os.path.join(temp_dir, "test_encoder.pt")
        
        history = trainer.train(
            train_data=train_data,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            save_path=encoder_path,
        )
        
        # Verify encoder was saved
        assert os.path.exists(encoder_path), "Encoder not saved"
        print(f"✓ Encoder saved to: {encoder_path}")
        
        # Verify loss decreased
        if len(history["train_loss"]) >= 2:
            print(f"✓ Training losses: {history['train_loss']}")
        
        # Load and verify
        from contrastive_encoder.models.encoder import KeystrokeEncoder
        encoder = KeystrokeEncoder()
        encoder.load_state_dict(torch.load(encoder_path))
        
        test_input = torch.randn(1, 11, 3)
        output = encoder(test_input)
        assert output.shape == (1, 128), "Loaded encoder output shape wrong"
        print("✓ Loaded encoder produces correct output")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\n✓ Quick training test PASSED")
    return True


def test_end_to_end():
    """Test end-to-end verification."""
    print("\n" + "="*60)
    print("TEST: End-to-End Verification")
    print("="*60)
    
    from contrastive_encoder.models.encoder import KeystrokeEncoder
    from contrastive_encoder.models.adapter import UserAdapter, verify_sample
    
    # Create mock components
    encoder = KeystrokeEncoder()
    adapter = UserAdapter()
    
    # Create mock centroid
    centroid = torch.randn(128)
    centroid = torch.nn.functional.normalize(centroid, dim=0)
    
    # Create sample
    sample = torch.randn(11, 3)
    
    # Verify
    result = verify_sample(
        encoder=encoder,
        adapter=adapter,
        sample=sample,
        centroid=centroid,
        threshold=0.5,
    )
    
    assert "accept" in result, "Missing 'accept' in result"
    assert "similarity" in result, "Missing 'similarity' in result"
    assert isinstance(result["accept"], bool), "accept should be bool"
    assert isinstance(result["similarity"], float), "similarity should be float"
    
    print(f"✓ Verification result: {result}")
    print("✓ End-to-end verification working")
    
    print("\n✓ End-to-end test PASSED")
    return True


def run_all_tests():
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("KEYSTROKE DYNAMICS - SMOKE TESTS")
    print("="*60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Encoder", test_encoder),
        ("Adapter", test_adapter),
        ("Quick Training", test_quick_training),
        ("End-to-End", test_end_to_end),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Run smoke tests")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "data", "encoder", "adapter", "training", "e2e"],
        help="Which test to run",
    )
    
    args = parser.parse_args()
    
    if args.test == "all":
        success = run_all_tests()
    elif args.test == "data":
        success = test_data_loading()
    elif args.test == "encoder":
        success = test_encoder()
    elif args.test == "adapter":
        success = test_adapter()
    elif args.test == "training":
        success = test_quick_training()
    elif args.test == "e2e":
        success = test_end_to_end()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
