import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QComboBox, QMessageBox, QHBoxLayout, QTextBrowser
)
from PyQt5.QtGui import QColor, QIcon, QFont
from PyQt5.QtCore import Qt

class RecipeRecommendationApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Recipe Recommendation System')
        self.setWindowIcon(QIcon('icon.png'))
        self.setGeometry(100, 100, 800, 500)

        self.title_label = QLabel('Recipe Recommendation System')
        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")

        self.team_label = QLabel('By: TeamByte<br>')
        self.team_label.setStyleSheet("font-size: 10px; margin-bottom: 10px;")

        self.label = QLabel('"Culinary Delights Await: Explore Exquisite Recipes Tailored Just for You!"')
        self.label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")

        self.setStyleSheet(
            "QWidget { background-color: #f5f5f5; }"
            "QLabel { font-weight: bold; }"
            "QTextEdit, QTextBrowser { background-color: #ffffff; }"
            "QPushButton { background-color: #4caf50; color: #ffffff; border: none; padding: 8px 12px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QComboBox { padding: 4px 8px; }"
            "QTextBrowser { background-color: #ffffff; border: 1px solid #d4d4d4; padding: 8px; }"
        )

        self.ingredients_label = QLabel('Enter Ingredients:')
        self.ingredients_input = QTextEdit()
        self.ingredients_input.textChanged.connect(self.validate_ingredients)
        self.recommend_button = QPushButton('Recommend')
        self.recommend_button.setEnabled(False)  # Disable initially
        self.recommend_button.clicked.connect(self.on_recommend_button_clicked)
        self.recommendation_output = QTextBrowser()
        self.recommendation_output.setReadOnly(True)

        self.num_recommendations_label = QLabel('Number of Recommendations:')
        self.num_recommendations_combobox = QComboBox()
        self.num_recommendations_combobox.addItem('1')
        self.num_recommendations_combobox.addItem('5')

        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_fields)

        # Create a layout for input area
        input_layout = QVBoxLayout()
        input_layout.setSpacing(15)  # Set spacing to 0
        input_layout.addWidget(self.ingredients_label)
        input_layout.addWidget(self.ingredients_input)

        num_recommendations_layout = QHBoxLayout()
        num_recommendations_layout.addWidget(self.num_recommendations_label)
        num_recommendations_layout.addWidget(self.num_recommendations_combobox)

        input_layout.addLayout(num_recommendations_layout)
        input_layout.addWidget(self.recommend_button)

        # Create a layout for output area
        output_layout = QVBoxLayout()
        output_layout.addWidget(self.recommendation_output)
        output_layout.addWidget(self.reset_button)

        # Create a main layout to hold input and output layouts
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.team_label)
        main_layout.addWidget(self.label)
        main_layout.addLayout(input_layout)
        main_layout.addLayout(output_layout)

        self.setLayout(main_layout)

        # Load the dataset and create or load the models
        try:
            self.data = pd.read_csv("full_dataset_0.csv")
            self.recipe_titles = self.data["title"]
            self.recipe_directions = self.data["directions"]
            self.content_vectorizer = TfidfVectorizer()
            self.content_matrix = self.content_vectorizer.fit_transform(self.data["ingredients"])

        except FileNotFoundError:
            self.show_error_message("Dataset file not found.")
        except Exception as e:
            self.show_error_message(str(e))

        self.apply_custom_style()

    def apply_custom_style(self):
        # Update font size and alignment for labels
        font = QFont()
        font.setPointSize(10)
        self.ingredients_label.setFont(font)
        self.num_recommendations_label.setFont(font)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.team_label.setAlignment(Qt.AlignCenter)

        # Add negative top margin to the ingredients label to reduce the space
        self.ingredients_label.setStyleSheet("margin-top: -5px;")

        # Increase font size for the input area
        input_font = QFont()
        input_font.setPointSize(10)
        self.ingredients_input.setFont(input_font)

        # Set fixed height for buttons
        self.recommend_button.setFixedHeight(30)
        self.reset_button.setFixedHeight(30)

        # Set expandable behavior for text areas
        self.ingredients_input.setLineWrapMode(QTextEdit.WidgetWidth)
        self.recommendation_output.setLineWrapMode(QTextEdit.WidgetWidth)
        self.ingredients_input.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.recommendation_output.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        output_font = QFont()
        output_font.setPointSize(10)
        self.recommendation_output.setFont(output_font)


        # Center-align the label
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")

        # Adjust output field height
        self.recommendation_output.setMinimumHeight(300)

    def content_based_recommendation(self, ingredients, num_recommendations):
        transformed_input = self.content_vectorizer.transform([ingredients])
        cosine_similarities = linear_kernel(transformed_input, self.content_matrix).flatten()
        related_indices = cosine_similarities.argsort()[:-num_recommendations - 1:-1]
        recommendations = []
        for index in related_indices:
            recipe_title = self.recipe_titles.iloc[index]
            recipe_directions = self.recipe_directions.iloc[index]
            recommendation = f"<strong>Recipe:</strong><br> {recipe_title}<br><strong>Directions:</strong><br> {recipe_directions}<br><br>"
            recommendations.append(recommendation)

        if num_recommendations == 1:
            recommendations.insert(0, "<strong>Recommended Recipe:</strong><br><br>")
        else:
            recommendations.insert(0, "<strong>Recommended Recipes:</strong><br><br>")

        return recommendations

    def validate_ingredients(self):
        ingredients = self.ingredients_input.toPlainText().strip()
        if len(ingredients.split(',')) >= 5:
            self.recommend_button.setEnabled(True)
            self.ingredients_input.setStyleSheet("")
        else:
            self.recommend_button.setEnabled(False)
            self.ingredients_input.setStyleSheet("border: 1px solid red;")

    def on_recommend_button_clicked(self):
        ingredients = self.ingredients_input.toPlainText().strip()
        if not ingredients:
            self.show_error_message("Please enter some ingredients.")
            return

        if len(ingredients.split(',')) < 5:
            self.show_error_message("Please enter at least 5 ingredients.")
            return

        num_recommendations = int(self.num_recommendations_combobox.currentText())

        try:
            recommendations = self.content_based_recommendation(ingredients, num_recommendations)

            recommendation_text = "\n".join(recommendations)
            self.recommendation_output.setText(recommendation_text)
        except Exception as e:
            self.show_error_message(str(e))

    def reset_fields(self):
        self.ingredients_input.clear()
        self.recommendation_output.clear()
        self.recommend_button.setEnabled(False)
        self.ingredients_input.setStyleSheet("")  # Reset the border style

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec_()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    recipe_app = RecipeRecommendationApp()
    recipe_app.show()
    sys.exit(app.exec_())
