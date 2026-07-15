import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = [
    {"title": "The Dark Knight", "tags": "batman superhero crime thriller action nolan gotham joker"},
    {"title": "Inception", "tags": "dream scifi thriller action mind heist nolan"},
    {"title": "Interstellar", "tags": "space scifi drama adventure nolan time wormhole"},
    {"title": "The Matrix", "tags": "scifi action cyberpunk virtual reality hacker dystopia"},
    {"title": "Avengers: Endgame", "tags": "superhero action marvel avengers infinity stones"},
    {"title": "Iron Man", "tags": "superhero action marvel tony stark technology"},
    {"title": "The Avengers", "tags": "superhero action marvel team heroes"},
    {"title": "Doctor Strange", "tags": "superhero action marvel magic multiverse"},
    {"title": "Spider-Man: No Way Home", "tags": "superhero action marvel spider multiverse"},
    {"title": "Pulp Fiction", "tags": "crime thriller drama tarantino nonlinear"},
    {"title": "Fight Club", "tags": "thriller drama psychology identity twist"},
    {"title": "The Shawshank Redemption", "tags": "drama prison hope friendship"},
    {"title": "Forrest Gump", "tags": "drama comedy history life journey"},
    {"title": "The Godfather", "tags": "crime drama mafia family power"},
    {"title": "Goodfellas", "tags": "crime drama mafia gangster"},
    {"title": "Schindler's List", "tags": "history drama war holocaust"},
    {"title": "The Lion King", "tags": "animation adventure drama family africa"},
    {"title": "Toy Story", "tags": "animation adventure comedy family toys"},
    {"title": "Finding Nemo", "tags": "animation adventure comedy family ocean fish"},
    {"title": "Up", "tags": "animation adventure drama family journey"},
    {"title": "WALL-E", "tags": "animation scifi romance environment future"},
    {"title": "Coco", "tags": "animation adventure family music mexico"},
    {"title": "Moana", "tags": "animation adventure family music ocean"},
    {"title": "Frozen", "tags": "animation adventure family musical ice"},
    {"title": "The Hunger Games", "tags": "action adventure dystopia scifi survival"},
    {"title": "Harry Potter and the Sorcerer's Stone", "tags": "fantasy adventure magic school wizard"},
    {"title": "The Lord of the Rings: The Fellowship of the Ring", "tags": "fantasy adventure action epic quest"},
    {"title": "The Hobbit: An Unexpected Journey", "tags": "fantasy adventure action epic quest dragon"},
    {"title": "Pirates of the Caribbean", "tags": "adventure action comedy fantasy pirate"},
    {"title": "Jurassic Park", "tags": "scifi adventure thriller dinosaur island"},
    {"title": "Titanic", "tags": "romance drama history disaster ship"},
    {"title": "Avatar", "tags": "scifi adventure action fantasy alien planet"},
    {"title": "Gravity", "tags": "scifi thriller drama space survival"},
    {"title": "The Martian", "tags": "scifi drama survival space mars"},
    {"title": "Mad Max: Fury Road", "tags": "action adventure scifi dystopia desert"},
    {"title": "John Wick", "tags": "action thriller crime assassin revenge"},
    {"title": "Mission: Impossible – Fallout", "tags": "action thriller spy adventure"},
    {"title": "Casino Royale", "tags": "action thriller spy bond adventure"},
    {"title": "The Bourne Identity", "tags": "action thriller spy amnesia"},
    {"title": "Gladiator", "tags": "action adventure drama history rome"},
    {"title": "Braveheart", "tags": "action adventure drama history scotland"},
    {"title": "300", "tags": "action adventure history war sparta"},
    {"title": "Troy", "tags": "action adventure drama history war greece"},
    {"title": "Kingdom of Heaven", "tags": "action adventure drama history crusades"},
    {"title": "Dunkirk", "tags": "action drama history war nolan"},
    {"title": "Saving Private Ryan", "tags": "action drama history war"},
    {"title": "Hacksaw Ridge", "tags": "action drama history war"},
    {"title": "1917", "tags": "action drama history war"},
    {"title": "The Revenant", "tags": "adventure drama survival western"},
    {"title": "No Country for Old Men", "tags": "crime drama thriller western"},
]

df = pd.DataFrame(movies_data)

cv = CountVectorizer(max_features=500, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()
similarity = cosine_similarity(vectors)

movies_dict = df[["title"]].to_dict()

with open("movies.pkl", "wb") as f:
    pickle.dump(movies_dict, f)
with open("similarity.pkl", "wb") as f:
    pickle.dump(similarity, f)

print(f"Saved movies.pkl ({len(df)} movies) and similarity.pkl ({similarity.shape})")
