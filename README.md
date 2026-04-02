RaceOutcomePred
Machine learning models to predict Formula 1 podium finishes using pre-race data.
The Prediction Task
Before a race starts, we feed the model a set of features — things we know ahead of time like grid position, qualifying position, and how the driver and constructor have been performing so far that season. The model outputs a probability that a given driver will finish in the top 3 (P1–P3).
Concrete example: Say it's the 2022 Bahrain GP and you want to predict whether Leclerc will podium:

He qualified P1, so grid = 1 and quali_position = 1
It's the first race of the season, so rolling stats are initialized to the season mean
The model compares these numbers against patterns learned from 2014–2021 and outputs a podium probability

Why it's not cheating: We only use information available before the race starts — no lap times, no live race positions, no DNF info. That's what makes it a genuine prediction rather than a lookup.
What the model learns: From 8 seasons of training data (2014–2021) it learns patterns like:

Starting from pole → high podium probability
Constructor has been strong this season → higher probability
Driver has been consistently finishing well → higher probability