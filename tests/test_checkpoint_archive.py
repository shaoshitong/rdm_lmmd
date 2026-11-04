import hashlib
import json
from pathlib import Path
from unittest import mock
import pytest
import importlib.util
spec = importlib.util.spec_from_file_location('checkpoint_archive_under_test',
    Path(__file__).resolve().parents[1]/'rdm/sw_lmmd/checkpoint_archive.py')
archive_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_module)
archive_checkpoint, wait_for_space = archive_module.archive_checkpoint, archive_module.wait_for_space


def validator(path, step):
    return dict(step=step, model_dtype='test-bytes')


def test_roundtrip_receipt_before_removing_source(tmp_path):
    source = tmp_path/'step_0000200.pth'; payload = bytes(range(256))*1024
    source.write_bytes(payload)
    proof = tmp_path/'control/proof.json'
    result = archive_checkpoint(source, tmp_path/'archive', proof, validator)
    assert not source.exists()
    assert Path(result['destination']).read_bytes() == payload
    assert json.loads(proof.read_text())['sha256'] == hashlib.sha256(payload).hexdigest()
    assert result['local_source_removed'] is True


def test_failed_readback_preserves_only_source_and_does_not_publish(tmp_path):
    source = tmp_path/'step_0000250.pth'; source.write_bytes(b'only-good-copy')
    with mock.patch.object(archive_module, 'sha256', return_value='corrupt'):
        with pytest.raises(AssertionError, match='readback'):
            archive_checkpoint(source, tmp_path/'archive', tmp_path/'proof.json', validator)
    assert source.read_bytes() == b'only-good-copy'
    assert not (tmp_path/'archive'/source.name).exists()
    assert not (tmp_path/'proof.json').exists()


def test_existing_different_checkpoint_is_never_overwritten(tmp_path):
    source = tmp_path/'step_0000300.pth'; source.write_bytes(b'new-unique')
    archive = tmp_path/'archive'; archive.mkdir(); target = archive/source.name; target.write_bytes(b'previous-unique')
    with pytest.raises(AssertionError, match='differs'):
        archive_checkpoint(source, archive, tmp_path/'proof.json', validator)
    assert source.read_bytes() == b'new-unique' and target.read_bytes() == b'previous-unique'


def test_resume_is_never_an_archive_deletion_candidate(tmp_path):
    source = tmp_path/'resume.pth'; source.write_bytes(b'state')
    with pytest.raises(AssertionError, match='Never'):
        archive_checkpoint(source, tmp_path/'archive', tmp_path/'proof.json', validator)
    assert source.exists()


def test_low_space_with_dead_archiver_stops_before_write(tmp_path):
    exitfile = tmp_path/'archive.exit'; exitfile.write_text('1\n')
    with mock.patch.object(archive_module.shutil, 'disk_usage', return_value=mock.Mock(free=10)):
        with pytest.raises(RuntimeError, match='Archive worker exited'):
            wait_for_space(tmp_path, 100, exitfile)


def test_failed_remote_flush_preserves_source(tmp_path):
    source = tmp_path/'step_0000350.pth'; source.write_bytes(b'only-unique-model')
    with mock.patch.object(archive_module.os, 'fsync', side_effect=OSError(28, 'No space left on device')):
        with pytest.raises(OSError):
            archive_checkpoint(source, tmp_path/'archive', tmp_path/'proof.json', validator)
    assert source.read_bytes() == b'only-unique-model'
    assert not (tmp_path/'archive'/source.name).exists()
    assert not (tmp_path/'proof.json').exists()
