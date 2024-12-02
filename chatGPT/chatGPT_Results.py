import matplotlib.pyplot as plt

# No adjustments
trainSupport = 19
trainOppose = 81
trainYes = 26
trainNo = 74

predictSupportStart = 8
predictOpposeStart = 92
predictYesStart = 3
predictNoStart = 97

accuracySupportStart = predictSupportStart / trainSupport
accuracyOpposeStart = predictOpposeStart / trainOppose
accuracyYesStart = predictYesStart / trainYes
accuracyNoStart = predictNoStart / trainNo

print("Accuracy for no adjustments")
print("-----------------------------")
print(f"Accuracy for Support: {accuracySupportStart:.4f}")
print(f"Accuracy for Oppose: {accuracyOpposeStart:.4f}")
print(f"Accuracy for Yes: {accuracyYesStart:.4f}")
print(f"Accuracy for No: {accuracyNoStart:.4f}")
print()

# Less Context
predictSupport = 2
predictOppose = 98
predictYes = 3
predictNo = 97

accuracySupportLC = predictSupport / trainSupport
accuracyOpposeLC = predictOppose / trainOppose
accuracyYesLC = predictYes / trainYes
accuracyNoLC = predictNo / trainNo

print("Accuracy for less context")
print("-----------------------------")
print(f"Accuracy for Support: {accuracySupportLC:.4f}")
print(f"Accuracy for Oppose: {accuracyOpposeLC:.4f}")
print(f"Accuracy for Yes: {accuracyYesLC:.4f}")
print(f"Accuracy for No: {accuracyNoLC:.4f}")
print()

# Samples
predictSupport = 9
predictOppose = 91
predictYes = 8
predictNo = 92

accuracySupportContext = predictSupport / trainSupport
accuracyOpposeContext = predictOppose / trainOppose
accuracyYesContext = predictYes / trainYes
accuracyNoContext = predictNo / trainNo

print("Accuracy for giving samples")
print("-----------------------------")
print(f"Accuracy for Support: {accuracySupportContext:.4f}")
print(f"Accuracy for Oppose: {accuracyOpposeContext:.4f}")
print(f"Accuracy for Yes: {accuracyYesContext:.4f}")
print(f"Accuracy for No: {accuracyNoContext:.4f}")
print()

# Explain
predictSupport = 6
predictOppose = 94
predictYes = 6
predictNo = 94

accuracySupportExplain = predictSupport / trainSupport
accuracyOpposeExplain = predictOppose / trainOppose
accuracyYesExplain = predictYes / trainYes
accuracyNoExplain = predictNo / trainNo

print("Accuracy for explaining")
print("-----------------------------")
print(f"Accuracy for Support: {accuracySupportExplain:.4f}")
print(f"Accuracy for Oppose: {accuracyOpposeExplain:.4f}")
print(f"Accuracy for Yes: {accuracyYesExplain:.4f}")
print(f"Accuracy for No: {accuracyNoExplain:.4f}")
print()

# plots
cats = ["Initial", "Less Context", "Samples", "Explain"]
supportArr = [accuracySupportStart, accuracySupportLC, accuracySupportContext, accuracySupportExplain]
plt.bar(cats, supportArr)
plt.title("Predicted Count Accuracy per Condition for Support")
plt.axhline(y=1, color='red', linestyle='--', label='y = 1')
plt.savefig("chatGPT/SupportPic.png")
plt.clf()

opposeArr = [accuracyOpposeStart, accuracyOpposeLC, accuracyOpposeContext, accuracyOpposeExplain]
plt.bar(cats, opposeArr)
plt.title("Predicted Count Accuracy per Condition for Oppose")
plt.axhline(y=1, color='red', linestyle='--', label='y = 1')
plt.savefig("chatGPT/OpposePic.png")
plt.clf()

yesArr = [accuracyYesStart, accuracyYesLC, accuracyYesContext, accuracyYesExplain]
plt.bar(cats, yesArr)
plt.title("Predicted Count Accuracy per Condition for Yes")
plt.axhline(y=1, color='red', linestyle='--', label='y = 1')
plt.savefig("chatGPT/YesPic.png")
plt.clf()

noArr = [accuracyNoStart, accuracyNoLC, accuracyNoContext, accuracyNoExplain]
plt.bar(cats, noArr)
plt.title("Predicted Count Accuracy per Condition for No")
plt.axhline(y=1, color='red', linestyle='--', label='y = 1')
plt.savefig("chatGPT/NoPic.png")
plt.clf()