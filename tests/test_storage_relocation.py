"""No source payload may be removed on a transfer/hash/readback failure."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('storage_relocation',Path(__file__).parents[1]/'ops/h20-storage-relocate.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'source';self.source.write_bytes(b'reference-data'*100)
        self.destination=self.root/'object/final';self.sha=hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.control=patch.object(m,'CONTROL',self.root/'control');self.control.start()
    def tearDown(self):
        self.control.stop();self.temp.cleanup()
    def transfer(self,digest=None):
        return m.transfer(self.source,self.destination,self.source.stat().st_size,digest or self.sha,'file.json')
    def test_success_final_name_readback_precedes_receipt(self):
        result=self.transfer();self.assertTrue(result['final_named_SHA_readback'])
        self.assertEqual(self.destination.read_bytes(),self.source.read_bytes())
        self.assertTrue((self.root/'control/files/file.json').exists())
    def test_wrong_source_hash_keeps_original_no_receipt(self):
        with self.assertRaises(AssertionError):self.transfer('0'*64)
        self.assertTrue(self.source.exists());self.assertFalse(self.destination.exists())
        self.assertFalse((self.root/'control/files/file.json').exists())
    def test_wrong_existing_object_never_overwritten(self):
        self.destination.parent.mkdir();self.destination.write_bytes(b'wrong')
        with self.assertRaises(AssertionError):self.transfer()
        self.assertTrue(self.source.exists());self.assertEqual(self.destination.read_bytes(),b'wrong')
    def test_final_readback_failure_retains_original(self):
        with patch.object(m,'sha',return_value='0'*64):
            with self.assertRaises(AssertionError):self.transfer()
        self.assertTrue(self.source.exists());self.assertFalse((self.root/'control/files/file.json').exists())

if __name__=='__main__':unittest.main()
