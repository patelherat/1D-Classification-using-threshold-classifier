import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def model(data, attribute):
    best_cost_func_yet = float('inf')
    # best_threshold_to_use = float('inf')
    idx = 0
    false_alarm_rate = []
    true_pos_rate = []
    cost = []
    thresh = []
    ind = []
    # for dc in data.columns:
    for i, attr in enumerate(attribute):
        TN = TP = FN = FP = 0
        for j, attr1 in enumerate(attribute):
            if attr1 > attr:                    # for every finds number of attributes which are lesser than threshold
                if data['Class'][j] == 1:       # if label is Assam
                    FN += 1
                else:
                    TN += 1
            else:
                if data['Class'][j] == 1:
                    TP += 1
                else:
                    FP += 1

        cost_fun = FN + FP                      # calculating the mistakes

        if cost_fun <= best_cost_func_yet:      # minimizing the mistakes
            best_cost_func_yet = cost_fun
            best_threshold_to_use = attr        # attribute used to classify
            best_idx = idx
            # best_attr = dc

        cost.append(cost_fun)
        thresh.append(attr)
        ind.append(idx)


        false_alarm_rate.append(FP / (TN + FP))
        true_pos_rate.append(TP / (FN + TP))
        idx += 1

    # print("Minimum Cost Function and Best Threshold and Best idx", best_cost_func_yet, best_threshold_to_use, best_idx)

    params = [best_cost_func_yet, best_threshold_to_use, false_alarm_rate, true_pos_rate, cost, thresh, ind]
    return params


def main():
    data = pd.read_csv(sys.argv[1])

    ages = data["Age"].tolist()
    heights = data["Ht"].tolist()
    # earLobes = data["EarLobes"].tolist()
    # tailLns = data["TailLn"].tolist()
    # hairLns = data["HairLn"].tolist()
    # bangLns = data["BangLn"].tolist()
    # reach = data["Reach"].tolist()

    ages2 = []
    for i, age in enumerate(ages):
        ages2.append(round(age / 2) * 2)  # quantize the snowfolks age into bins(of 2 years)

    heights2 = []
    for i, height in enumerate(heights):
        heights2.append(round(height / 5) * 5)  # quantize the snowfolks age into bins(of 2 years)

    # earLobes2 = []
    # for i, el in enumerate(earLobes):
    #     earLobes2.append(round(el / 1) * 1)
    #
    # tailLns2 = []
    # for i, tail in enumerate(tailLns):
    #     tailLns2.append(round(tail / 1) * 1)
    #
    # hairLns2 = []
    # for i, hair in enumerate(hairLns):
    #     hairLns2.append(round(hair / 1) * 1)
    #
    # bangLns2 = []
    # for i, bang in enumerate(bangLns):
    #     bangLns2.append(round(bang / 1) * 1)
    #
    # reach2 = []
    # for i, rea in enumerate(reach):
    #     reach2.append(round(rea / 5) * 5)

    print("Training...")
    parameters1 = model(data, ages2)
    parameters2 = model(data, heights2)

    # checks which attribute gives lower cost function
    if parameters1[0] < parameters2[0]:
        xx = []
        yy = []
        for i, mp in enumerate(parameters1[4]):
            if mp == min(parameters1[4]):
                xx.append(parameters1[2][i])
                yy.append(parameters1[3][i])
                break

        # sorts according to threshold value
        xs, ys = zip(*sorted(zip(parameters1[5], parameters1[4])))

        plt.plot(xs, ys)
        plt.plot(parameters1[1], parameters1[0], 'ro')      # cost function vs threshold(age)
        plt.ylabel("cost function")
        plt.xlabel("threshold")
        plt.title("Age")
        plt.show()

        print("Model Trained")
        print("Minimum cost function is", parameters1[0])
        print("Age to classify Assam and Bhutan abominable snowfolks is", parameters1[1])

        parameters1[2].sort()
        parameters1[3].sort()
        plt.plot(parameters1[2], parameters1[3])        # ROC curve
        plt.plot(xx, yy, 'ro')
        plt.ylabel("TPR")
        plt.xlabel("FAR")
        plt.title("Age")
        plt.show()

        auc = np.trapz(parameters1[3], parameters1[2])
        print("Area under ROC curve is", auc)

        file_content = open("HW_02_Patel_Herat_Trained.py", "wt")
        file_content.write("x = ")
        file_content.write("%s" % (parameters1[1]))
        file_content.write("\n")
        # file_content.write("attribute = ")
        # file_content.write("%s" %(ages2))
        # file_content.write("\n")
        file_content.write("import pandas as pd\n")
        file_content.write("import sys\n")
        file_content.write("data = pd.read_csv(sys.argv[1])\n")
        file_content.write("ag = data[\"Age\"].tolist()\n")

        file_content.write("output = []\n")
        file_content.write("for h in ag:\n")
        file_content.write("\tif h > x:\n")
        file_content.write("\t\toutput.append(1)\n")        # if correctly classified(Assam)
        file_content.write("\telse:\n")
        file_content.write("\t\toutput.append(-1)\n")       # it's Bhutan

        file_content.write("out = pd.DataFrame(output, columns=['Predicted output'])\n")
        file_content.write("out.to_csv('HW_02_Patel_Herat_results.csv', header=True, index=False)\n")

        file_content.write("print(\"Output saved to csv file\")\n")

        # classify(ages2, file_content)

    else:
        # print(parameters2[4])
        xx = []
        yy = []
        for i, mp in enumerate(parameters2[4]):
            if mp == min(parameters2[4]):
                xx.append(parameters2[2][i])
                yy.append(parameters2[3][i])
                break
                # plt.plot(parameters2[2][i], parameters2[3][i], 'ro')
        #parameters2[5].sort()
        # parameters2[4].sort()
        xs, ys = zip(*sorted(zip(parameters2[5], parameters2[4])))

        plt.plot(xs, ys)
        plt.plot(parameters2[1], parameters2[0], 'ro')
        plt.ylabel("cost function")
        plt.xlabel("threshold")
        plt.title("Height")
        plt.show()

        print("Model Trained")
        print("Minimum cost function is", parameters2[0])
        print("Height to classify Assam and Bhutan abominable snowfolks is", parameters2[1])

        parameters2[2].sort()
        parameters2[3].sort()
        plt.plot(parameters2[2], parameters2[3])        # plotting ROC curve
        plt.plot(xx, yy, 'ro')
        plt.ylabel("TPR")
        plt.xlabel("FAR")
        plt.title("Height")
        plt.show()

        auc = np.trapz(parameters2[3], parameters2[2])      # area under curve
        print("Area under ROC curve is", auc)

        file_content = open("HW_02_Patel_Herat_Trained.py", "wt")
        file_content.write("x = ")                          # trained threshold value(attribute)
        file_content.write("%s" %(parameters2[1]))
        file_content.write("\n")
        # file_content.write("attribute = ")
        # file_content.write("%s" %(heights2))
        # file_content.write("\n")
        file_content.write("import pandas as pd\n")
        file_content.write("import sys\n")
        file_content.write("data = pd.read_csv(sys.argv[1])\n")
        file_content.write("heig = data[\"Ht\"].tolist()\n")

        file_content.write("output = []\n")
        file_content.write("for h in heig:\n")              # classifies the validation set
        file_content.write("\tif h > x:\n")
        file_content.write("\t\toutput.append(1)\n")
        file_content.write("\telse:\n")
        file_content.write("\t\toutput.append(-1)\n")

        file_content.write("out = pd.DataFrame(output, columns=['Predicted output'])\n")
        file_content.write("out.to_csv('HW_02_Patel_Herat_results.csv', index=False, header=True)\n")     # saves as csv file

        file_content.write("print(\"Output saved to csv file\")\n")

        # classify(heights2, file_content)


if __name__ == '__main__':
    main()
