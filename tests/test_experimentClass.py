from src.lib.experimentClass import ExpSetup
import unittest
from unittest.mock import patch
import os
import deepinv as dinv
from deepinv.optim.dpir import get_DPIR_params
from deepinv.optim.prior import PnP
from deepinv.optim.data_fidelity import L2
import torch



class TestExpSetup(unittest.TestCase):
    

    def setUp(self):
        self.exp=ExpSetup()
        hqs = "HQS"
        admm = "ADMM"
        
        data_fidelity = L2()
        prior = dinv.optim.prior.TVPrior(n_it_max=300)

        verbose = False
        plot_metrics = False

        stepsize = 1
        lamb = 1e-4  
        params_algo = {"stepsize": stepsize, "lambda": lamb}
        max_iter = 300
        early_stop = True
        
        self.exp.create_optimizer(hqs, hqs, prior=prior, data_fidelity=data_fidelity, early_stop=early_stop,
                             max_iter=max_iter, params_algo=params_algo)
        self.exp.create_optimizer(admm, admm, prior=prior, data_fidelity=data_fidelity, early_stop=early_stop,
                             max_iter=max_iter, params_algo=params_algo)    

    def test_init_no_args(self):
        exp=self.exp
        work_dir=os.getcwd()
        self.assertEqual(exp.work_dir, work_dir)
        self.assertEqual(exp.data_dir, work_dir +'/' + 'data')
        self.assertEqual(exp.src_dir, work_dir + '/' + 'src')
        self.assertEqual(exp.model_dir, work_dir +'/' + 'models')
        self.assertEqual(exp.out_dir, work_dir + '/' +'output')

    def test_init_work_dir_args(self):
        work_dir = "test_workdir"
        exp=ExpSetup(work_dir)
        self.assertEqual(exp.work_dir, work_dir)
        self.assertEqual(exp.data_dir, work_dir +'/' + 'data')
        self.assertEqual(exp.src_dir, work_dir + '/' + 'src')
        self.assertEqual(exp.model_dir, work_dir +'/' + 'models')
        self.assertEqual(exp.out_dir, work_dir + '/' +'output')
    
    def test_init_all_args(self):
        work_dir = "work_test"
        src_dir = "source_test"
        data_dir = "data_test"
        model_dir = "model_test"
        output_dir = "output_test"

        exp=ExpSetup(work_dir=work_dir, src_dir=src_dir, data_dir=data_dir,
                    model_dir=model_dir, out_dir=output_dir)
        
        self.assertEqual(exp.work_dir, work_dir)
        self.assertEqual(exp.data_dir, data_dir)
        self.assertEqual(exp.src_dir, src_dir)
        self.assertEqual(exp.model_dir, model_dir)
        self.assertEqual(exp.out_dir, output_dir)
    
    @patch('builtins.print')
    def test_optimizer_not_existing(self, mock_print):
        exp = self.exp
        # exp.create_model("hf-hub:BVRA/MegaDescriptor-T-224")
        model_name="hf-hub:BVRA/MegaDescriptor-T-224"
        self.assertIsNone(exp.create_model(model_name))
        mock_print.assert_called_with('The model path:', exp.model_dir+'/'+model_name, ' does not exist. Initializing with random weights.')
    
    def test_optimizer_download(self):
        exp = self.exp
        model_name="hf-hub:BVRA/MegaDescriptor-T-224"
        self.assertIsNotNone(exp.create_model(model_name, pretrained=True))

    @patch('builtins.print')
    def test_create_optimizer(self, mock_print):
        exp = self.exp
        hqs = "HQS"
        admm = "ADMM"
        
  
        self.assertIsNotNone(exp.get_optimizer(hqs))
        self.assertIsNotNone(exp.get_optimizer(admm))
        test_name = "TEST_OPTIM"
        self.assertIsNone(exp.get_optimizer(test_name))
        mock_print.assert_called_with('The optimizer ', test_name, ' has not been created')

 

if __name__=='__main__':
	unittest.main()
