# MedChat AI
This project implements an AI-powered chatbot specializing in healthcare information. It uses a vector database for efficient information retrieval and a language model for generating responses.

## Features
- Answers medical questions using a combination of pre-stored information and AI-generated responses
- Uses a vector database (Chroma) for efficient information retrieval
- Falls back to a base language model when no relevant information is found in the database
- Preprocesses and splits healthcare data for training and validation

## Data Source
This project uses the Comprehensive Medical Q&A Dataset from Kaggle:
[Comprehensive Medical Q&A Dataset](https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset/data)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
We gratefully acknowledge the use of the Comprehensive Medical Q&A Dataset from Kaggle in this project.

## TODO
- [ ] Perform more extensive data exploration
   - [ ] Analyze distribution of question types
   - [ ] Examine question and answer lengths
   - [ ] Identify common medical terms and topics
   - [ ] Visualize key insights from the dataset

- [ ] Enhance data preprocessing
   - [ ] Implement text cleaning (remove special characters, standardize formatting)
   - [ ] Perform named entity recognition for medical terms
   - [ ] Apply advanced tokenization techniques
   - [ ] Explore options for data augmentation

- [ ] Evaluate and improve data quality
   - [ ] Identify and handle potential data inconsistencies
   - [ ] Implement a method to detect and remove duplicate entries
   - [ ] Consider adding additional metadata (e.g., difficulty level, topic categories)

- [ ] Optimize data storage and retrieval
   - [ ] Experiment with different vector embedding techniques
   - [ ] Fine-tune Chroma database parameters for better performance
   - [ ] Implement caching mechanisms for frequently accessed data

- [ ] Enhance model performance and capabilities
   - [ ] Experiment with fine-tuning the language model on medical data
   - [ ] Implement a hybrid retrieval-generation approach
   - [ ] Explore few-shot learning techniques for improved accuracy

- [ ] Improve user experience
   - [ ] Develop a user-friendly web interface or chat application
   - [ ] Implement multi-turn conversations and context tracking
   - [ ] Add support for voice input and text-to-speech output

