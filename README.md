Run SVM  Decision Tree
1   0.99    0.9
2   0.98    0.87
3   0.99    0.88
4   0.98    0.87
5   0.99    0.87

mean:  0.99   0.88
std:  0.01   0.01



docker build -t exp:v1 -f docker/Dockerfile .

docker run -it exp:v1

export FLASK_APP=api/app.py ; flask run