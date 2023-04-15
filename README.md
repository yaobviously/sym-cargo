# Synmax Competition 2021 Sponsed by Bill Perkins (@bp22)

This repository contains code and data for a project completed as part of a Twitter competition in 2021. The objective was to reconstruct each vessel's voyages and predict their next three ports.

# Data
The data was provided by [Synmax](https://www.synmax.com/) in two CSV files. The first file contained the location of the ports, while the second file contained irregular tracking data indicating the x, y coordinates, speed, and heading of each vessel.

# Methodology
To reconstruct the voyages I used a k-nearest neighbors to identify when ships were close to ports. Since the earth is round and not flat, haversine distance was used. But since the earth is not perfectly round and some of the ports were very close together, haversine distance wasn't precise enough. I had to use vincenty distance to get sufficiently accurate labels.

There were other nuances involved in figuring out exactly when a ship was in and out of port because the contest required accuracy to within 24 hours. See the notebook for details. 

The predictive model for determining the next three ports of the vessels relied on two steps. First, the DBSCAN clustering algorithm was used to group ports together. Second, a KMeans model clustered the vessels. The predictive features were the DBSCAN cluster labels of the last three ports and the KMeans cluster label. Three separate XGBClassifier models were trained to predict the ports one, two, and three steps ahead, with the second most likely port being selected if the predicted ports matched.

# Results

I ended up winning the competition and $10,000. 

# Comments

The original notebook was done in Google Colab. It was messy. To address this I created a new notebook and put the functions into a separate script to make it easier to look over the code. A few changes were made, and unneeded code was discarded, but the essence of the project remains. I would do it differently (and better!) in 2023 but I'm still proud of putting it all together.