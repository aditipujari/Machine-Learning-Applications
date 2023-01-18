# Always encode string with integer
# Binary Classification example
# Import required library
from sklearn import tree

# Load the dataset
# Rough  1
# Smooth 0(hot)      1-hot encoding

# Tennis 1
# Cricket 2

def BallPredictor(weight,surface):
    Features = [[35, 1], [47, 1], [90, 0], [48, 1], [90, 0], [35, 1], [92, 0], [35, 1], [35, 1], [35, 1], [96, 0],
                [43, 1], [110, 0], [35, 1], [95, 0]]
    Labels = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2]

    # Decide the ML Algorithm
    obj = tree.DecisionTreeClassifier()

    # Perform the training of model
    obj = obj.fit(Features, Labels)  # paramters - 1.List of Features 2. List of Labels

    # Perform the testing
    ret = obj.predict([[weight, surface]])
    if ret == 1:
        print("Your object looks like TENNIS ball")
    else:
        print("Your object looks like CRICKET ball")

def main():
    print("-------------Ball Predictor Case Study-------------")
    print("Please enter the weight of your object in grams")
    weight = int(input())

    print("Please enter the tpe of surface of your object(Rough / Smooth)")
    surface = input()

    if surface.lower() == "rough":
        surface = 1
    elif surface.lower() == "smooth":
        surface = 0
    else:
        print("Invalid type of surface")
        exit()

    BallPredictor(weight,surface)
    
if __name__ == "__main__":
    main()


