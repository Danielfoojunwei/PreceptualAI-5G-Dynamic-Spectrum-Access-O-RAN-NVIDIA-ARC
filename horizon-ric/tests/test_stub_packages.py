"""Implemented packages have a real public API — federated, trading,
encoder are now built out (Tier-1 in the encoder case)."""


def test_implemented_packages_have_public_api():
    from horizon_ric.encoder import (
        AssetAttributes,
        EntityTokenizer,
        EntityType,
        SpatialPrior,
    )
    from horizon_ric.federated import (
        ClientUpdate,
        FedAvg,
        FedProx,
        top_k_sparsify,
    )
    from horizon_ric.trading import (
        Auctioneer,
        Bidder,
        SealedBidAuction,
    )

    assert callable(FedAvg)
    assert callable(FedProx)
    assert callable(top_k_sparsify)
    assert callable(Auctioneer)
    assert callable(Bidder)
    assert callable(SealedBidAuction)
    assert callable(SpatialPrior)
    assert callable(EntityTokenizer)
    assert ClientUpdate.__name__ == "ClientUpdate"
    assert EntityType.UE.value == "ue"
    assert AssetAttributes.__name__ == "AssetAttributes"


def test_top_level_package_advertises_only_implemented():
    import horizon_ric

    for name in ("DecisionRecord", "HorizonRAppLifecycle", "RAppState",
                 "JsonlEvidenceStore", "SqliteEvidenceStore",
                 "generate_human_explanation"):
        assert hasattr(horizon_ric, name), f"missing public name: {name}"
    assert horizon_ric.__version__ == "0.1.0"
