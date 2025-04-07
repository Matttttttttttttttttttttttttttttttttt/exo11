import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score, f1_score, precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('web-page-phishing.csv')

WHITELIST = {"https://www.google.com/", "https://www.amazon.com/","https://www.facebook.com/","https://www.apple.com/","https://www.microsoft.com/","https://www.netflix.com/",
    "https://www.linkedin.com/","https://www.twitter.com/","https://www.instagram.com/","https://www.wikipedia.org/","https://www.reddit.com/","https://www.yahoo.com/",
    "https://www.ebay.com/","https://www.paypal.com/","https://www.bing.com/","https://www.cnn.com/","https://www.nytimes.com/","https://www.wsj.com/","https://www.bbc.com/",
    "https://www.forbes.com/","https://www.nationalgeographic.com/","https://www.imdb.com/","https://www.quora.com/","https://www.github.com/","https://www.stackoverflow.com/",
    "https://www.medium.com/","https://www.adobe.com/","https://www.shopify.com/","https://www.airbnb.com/","https://www.uber.com/","https://www.lyft.com/","https://www.dropbox.com/",
    "https://www.slack.com/","https://www.tiktok.com/","https://www.spotify.com/","https://www.paytm.com/","https://www.tumblr.com/","https://www.flickr.com/","https://www.soundcloud.com/",
    "https://www.vimeo.com/","https://www.wordpress.com/","https://www.nike.com/","https://www.adidas.com/","https://www.coursera.org/","https://www.edx.org/","https://www.udemy.com/",
    "https://www.khanacademy.org/","https://www.oracle.com/","https://www.ibm.com/","https://www.intel.com/","https://www.samsung.com/","https://www.hulu.com/","https://www.disneyplus.com/",
    "https://www.snapchat.com/","https://www.pinterest.com/","https://www.twitch.tv/"
}

df_features = ['url_length', 'n_dots', 'n_hypens', 'n_underline', 'n_slash',
       'n_questionmark', 'n_equal', 'n_at', 'n_and', 'n_exclamation',
       'n_space', 'n_tilde', 'n_comma', 'n_plus', 'n_asterisk', 'n_hastag',
       'n_dollar', 'n_percent', 'n_redirection']

def analyzeurl(userurl):
#compte le nb de caractères, puis tout les caractères spéciaux comptés
    charcount = [ userurl.count("."), userurl.count("-"), userurl.count("_"), userurl.count("/"), userurl.count("?"), userurl.count("="), userurl.count("@"),
    userurl.count("&"), userurl.count("!"), userurl.count(" "), userurl.count("~"), userurl.count(","), userurl.count("+"), userurl.count("*"),
    userurl.count("#"), userurl.count("$"), userurl.count("%") ]
    secure_protocol  = 1 if ("https://" in userurl or "http://" in userurl) else 0

    features = [ [len(userurl)] + charcount + [secure_protocol] ]

    return features
#on dit au modèle de prendre en considération en plus si l'url commence pas avec https, et si il contiens un des mots les plus commun dans les liens de phishing


x = data.drop("phishing", axis=1)
y = data["phishing"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(x_train)
xtest_scaled = scaler.transform(x_test)

model = LogisticRegression(max_iter=2000)
model.fit(xtrain_scaled, y_train)

y_pred = model.predict(xtest_scaled)
y_pred_proba = model.predict_proba(xtest_scaled)[:,1]
print(f"model accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

custom_thresh = 0.32
y_pred_custom = (y_pred_proba >= custom_thresh).astype(int)

userurl = str(input("Enter your URL : "))

urlfeatures = analyzeurl(userurl)
#reshape en array 2D numpy
urlfreshaped = np.array(urlfeatures).reshape(1,-1)

urlfeaturesdf = pd.DataFrame(urlfreshaped, columns=df_features)

scaled_features = scaler.transform(urlfeaturesdf)

probas = model.predict_proba(scaled_features)[0][1]

if userurl in WHITELIST:
    print("Trusted URL, contained in whitelist")
else:
    print(f"Il y a {probas * 100:.2f}% de chances que votre lien soit un lien de phishing")
