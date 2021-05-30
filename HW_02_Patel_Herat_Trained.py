x = 44
import pandas as pd
import sys
data = pd.read_csv(sys.argv[1])
ag = data["Age"].tolist()
output = []
for h in ag:
	if h > x:
		output.append(1)
	else:
		output.append(-1)
out = pd.DataFrame(output, columns=['Predicted output'])
out.to_csv('HW_02_Patel_Herat_results.csv', header=True, index=False)
print("Output saved to csv file")
