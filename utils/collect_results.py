import torch
import pandas as pd
import os
import traceback

"""Set location, experiment and metrics you want to collect"""
LOGDIR = "logs"
EXPERIMENT = "stanford_synthetic"
COLUMNS = ["version"]
METRICS = [ "ade_test", "fde_test", "feasibility_test"]



def create_results_pd(EXPERIMENT = EXPERIMENT,
					  COLUMNS = COLUMNS,
					  METRICS = METRICS,
					  LOGDIR = LOGDIR,
					  make_csv = False):
	"""
	Reads results from 'results.pt' checkpoints and writes them into a csv in 'resultCSV'
	"""
	result_pd = pd.DataFrame(columns = COLUMNS+METRICS)

	experiment_dir = os.path.join( LOGDIR, EXPERIMENT)
	versions = os.listdir(experiment_dir)
	for v in versions:
		try:
			res = torch.load(os.path.join(experiment_dir, v, "results.pt"), map_location='cpu')
			res_list = [v]
			print("file exists")
			for m in METRICS:
				res_list.append( res[m])



			result_pd.loc[len(result_pd)] = res_list


		except:
			pass


	if make_csv:
		result_pd.to_csv("resultCSV/{}.csv".format(EXPERIMENT))

if __name__ == "__main__":
	create_results_pd(make_csv=True)