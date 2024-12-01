import random
import json
import pickle
import numpy as np
import nltk
import math
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

class chatbotAI:
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = json.loads(open('Scripts\Chatbot\intents.json', encoding="utf8").read())

        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
        self.model = load_model('chatbot_model.h5')

        self.stateSpace = ["default", "typing coeff"]

        self.state = "default"

        self.inputNum = None
        self.arr = None
        self.current_coeffsNum = 0

        print("GO! Bot is running!")


    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words (self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class (self, sentence):
        bow = self.bag_of_words (sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes [r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intents_list, intents_json):
        print(intents_list)
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice (i['responses'])
                break
        return result


    def solveEq(self, arr):
        if(len(arr) == 2):
            a = float(arr[0])
            b = float(arr[1])
            if(a == 0 and b != 0):
                return "Phương trình vô nghiệm"
            elif(a == 0 and b == 0):
                return "Phương trình vô số nghiệm"
            else:
                return "Giá trị của biểu thức là %f" % (-b/a)
        elif(len(arr) == 3):
            a = float(arr[0])
            b = float(arr[1])     
            c = float(arr[2])
            delta = pow(b,2) - 4*a*c
            x1 = float(-b - math.sqrt(abs(delta)))/(2*a)
            x2 = float(-b + math.sqrt(abs(delta)))/(2*a)
            if(a == 0):
                return "Phương trình được cho không phải bậc 2, hãy thử lại sau"
            elif(delta < 0):
                xphuc1 = float(math.sqrt(abs((math.pow(b,2)/(4*a))-c)/a))
                xphuc2 = -xphuc1
                x3 = float((b)/(2*a))
                return "Phương trình có 2 nghiệm phức là: %fi - %f và %fi - %f" % (xphuc1, x3 , xphuc2 , x3)
            elif(delta == 0):
                return "Phương trình có nghiệm kép %f" %(-b/(2*a))
            else:
                return "Phương trình có 2 nghiệm là %f và %f"
        else:
            a = float(arr[0])
            b = float(arr[1])
            c = float(arr[2])
            d = float(arr[3])
            if(a == 0):
                return "Phương trình được cho không phải bậc 3, hãy thử lại sau"
            else:
                print("Đầu tiên, ta cần đưa phương trình trên về dạng chính tắc : y^3+ py+ q = 0, từ đó áp dụng công thức Cardano cho phương trình bậc 3")
            p = c/a - (b**2)/((a**2)*3)
            q = d/a+ ((2*b**3)-(9*a*b*c))/(27*(a**3))
            delta = ((p**3)/27) +((q**2)/4)                 
            if p == 0 and q == 0 and delta == 0:            # All 3 Roots are Real and Equal 
                if (d / a) >= 0:
                    x = (d / (1.0 * a)) ** (1 / 3.0) * -1
                else:
                    x = (-d / (1.0 * a)) ** (1 / 3.0)
                return "Phương trình chỉ có 1 nghiệm là %f" %(x)              

            elif delta < 0:                                # All 3 roots are Real

                i = math.sqrt(((q ** 2.0) / 4.0) - delta)   # Helper Temporary Variable
                j = i ** (1 / 3.0)                      # Helper Temporary Variable
                k = math.acos(-(q / (2 * i)))           # Helper Temporary Variable
                L = j * -1                              # Helper Temporary Variable
                M = math.cos(k / 3.0)                   # Helper Temporary Variable
                N = math.sqrt(3) * math.sin(k / 3.0)    # Helper Temporary Variable
                P = (b / (3.0 * a)) * -1                # Helper Temporary Variable

                x1 = 2 * j * math.cos(k / 3.0) - (b / (3.0 * a))
                x2 = L * (M + N) + P
                x3 = L * (M - N) + P

                return "Phương trình có 3 nghiệm là %f %f %f" %(x1, x2, x3)          # Returning Real Roots as numpy array.

            elif delta > 0:                                 # One Real Root and two Complex Roots
                R = -(q / 2.0) + math.sqrt(delta)           # Helper Temporary Variable
                if R >= 0:
                    S = R ** (1 / 3.0)                  # Helper Temporary Variable
                else:
                    S = (-R) ** (1 / 3.0) * -1          # Helper Temporary Variable
                    T = -(q / 2.0) - math.sqrt(delta)
                    if T >= 0:
                        U = (T ** (1 / 3.0))                # Helper Temporary Variable
                    else:
                        U = ((-T) ** (1 / 3.0)) * -1        # Helper Temporary Variable

                x1 = (S + U) - (b / (3.0 * a))
                x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * math.sqrt(3) * 0.5j
                x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * math.sqrt(3) * 0.5j

                return (f"Phương trình gồm 1 nghiệm thực và 2 nghiệm phức là {str(x1)}, {str(x2)}, {str(x3)}")           # Returning One Real Root and two Complex Roots as numpy array.


    def chatAPI(self, msg):
        intents_list = self.predict_class (msg)
        tag = intents_list[0]['intent']
        inputNumCases = ["math1", "math2", "math3"]

        match self.stateSpace.index(self.state)+1:
            case 1:
                
                # handle state transition
                if(tag in inputNumCases):
                    
                    self.inputNum = inputNumCases.index(tag) + 2
                    self.arr = np.zeros(self.inputNum)

                    # assign new state
                    self.state = self.stateSpace[ self.stateSpace.index(self.state)+1 ]
                    return "điền hệ số đi bro"

                else:
                    ints = self.predict_class (msg)
                    res = self.get_response (ints, self.intents)
                    return res

            case 2:
                if self.current_coeffsNum < self.inputNum:

                    self.arr[self.current_coeffsNum] = int(msg)

                    self.current_coeffsNum = self.current_coeffsNum + 1


                if self.current_coeffsNum == self.inputNum:
               
                    # assign new state
                    self.state = self.stateSpace[ self.stateSpace.index(self.state)-1 ]
                    
                    return self.solveEq(self.arr)
                else:
                    return "Nhập tiếp đi bro"

            case _:
                res = self.get_response (ints, self.intents)
                return res
