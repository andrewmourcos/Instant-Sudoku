# Instant-Sudoku
### The web app is now available here: https://instant-sudoku.herokuapp.com

Try taking a photo of a sudoku puzzle and use the script in this repo or the *webapp* to solve the puzzle automatically.

Through this project, I got to experiment with various classification techniques (sklearn), computer vision (OpenCV), backend web development (flask), server deployment (Heroku), etc.

I experimented with 3 different types of ML classifier objects: KNN, Logistic Regression and Support Vector Machines and I found that the logistic regression was the most accurate at identifying the digits in the puzzle. 

In the future I plan to create an instructions page for the webapp as well as a form to allow the user to correct errors made by the app. 

#### Tips: it is assumed that your sudoku puzzle is the largest object in the image (if not, you should crop it before uploading). Also, try to avoid shadows for an accurate recognition

