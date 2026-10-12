# Log guide

These files are the terminal output saved during the Task 1 experiment. They use the worksheet's original script names because the source files were renamed later.

| File | What it shows |
|---|---|
| `01-build-model.txt` | Training the MNIST model and checking its accuracy. |
| `02-convert-to-tflite.txt` | Converting the model and printing its file size. |
| `03-export-saved-model.txt` | Saving the model as a SavedModel folder and checking its size. |
| `04-local-inference.txt` | Testing the model on this computer and measuring prediction time. |
| `05-cloud-server.txt` | Starting the local web server. |
| `06-cloud-client.txt` | Sending test images to the server and measuring request time and data size. |
| `07-compare-results.txt` | Comparing local predictions with server predictions. |
| `08-precision-audit.txt` | Summary of the full-precision timing check. |
| `09-cloud-client-server-down.txt` | Showing that the client cannot use the server after it is stopped. |
| `10-local-after-server-down.txt` | Showing that local prediction still works after the server is stopped. |

`precision-comparison-samples.json` contains the individual timing values from 100 runs, along with their summary. It lets the reported averages and percentiles be checked.

The terminal logs have been kept as captured. Some may show colours, overwritten progress lines, or other control characters. The tables in `Task1_Report.md` give the same results in a cleaner form.
