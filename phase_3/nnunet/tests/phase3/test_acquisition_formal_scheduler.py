import acquisition_formal_scheduler as scheduler


def test_all_five_audited_smoke_runs_are_required(tmp_path):
    for arm in scheduler.ARMS:
        smoke = tmp_path / arm / "smoke"
        smoke.mkdir(parents=True)
        (smoke / "passed.json").write_text('{"passed": true}')
        (smoke / "checkpoint_audit.json").write_text('{"complete": true}')
        (smoke / "results_audit.json").write_text('{"complete": true}')
    assert scheduler.smoke_ready(tmp_path)
    (tmp_path / "a60_n2/smoke/results_audit.json").unlink()
    assert not scheduler.smoke_ready(tmp_path)
    assert scheduler.smoke_passed(tmp_path, "a60_n3")
    assert not scheduler.smoke_passed(tmp_path, "a60_n2")


def test_available_gpus_uses_free_cards_and_reservations(monkeypatch):
    lines = "0, 9407\n1, 15\n2, 21821\n3, 5037\n4, 15\n5, 15\n"
    monkeypatch.setattr(scheduler.subprocess, "check_output", lambda *args, **kwargs: lines)
    assert scheduler.available_gpus(set()) == [4, 5, 1]
    assert scheduler.available_gpus({1}) == [4, 5]
    assert scheduler.gpu_memory_mib()[2] == 21821
