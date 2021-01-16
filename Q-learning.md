Implementation of Q-learning model
================
  
Based on https://en.wikipedia.org/wiki/Q-learning


## Setup environment 

#### Clear workspace
``` r
rm(list = ls())
```

#### Load packages
``` r
library(tidyverse)
```

## Simulate data 

#### Create data frame for simulated data
``` r
n_trials <- 1000
states <- c("s1", "s2", "s3")
actions <- c("a1", "a2", "a3")

data <- tibble(
  state = factor(sample(states, n_trials, replace = TRUE)),
  action = NA,
  reward = NA,
  next_state = factor(dplyr::lead(state, 1))
)
head(data)
```

#### Simulate probabilistic responses and calculate rewards
``` r
data <- data %>% 
  mutate(action = factor(ifelse((state == "s1"), 
                                sample(actions, 
                                       length(data$action[data$state == "s1"]), 
                                       replace = TRUE, 
                                       prob = c(0.8, 0.1, 0.1)), 
                                ifelse((state == "s2"), 
                                       sample(actions, 
                                              length(data$action[data$state == "s2"]), 
                                              replace = TRUE, 
                                              prob = c(0.1, 0.8, 0.1)),
                                       sample(actions, 
                                              length(data$action[data$state == "s3"]), 
                                              replace = TRUE, 
                                              prob = c(0.1, 0.1, 0.8))))),
         reward = as.numeric(ifelse((state == "s1" & action == "a1")|
                                      (state == "s2" & action == "a2")|
                                      (state == "s3" & action == "a3"), 1, -1)))
head(data)
```

#### Confusion matrix
``` r
table(data$state, data$action)
prop.table(table(data$reward))
```

## Learn 

#### Q-learning function
``` r
Q_learn <- function(data, alpha, gamma) {
  # Inputs:
  # data  = data frame with columns 'state/action/reward/next_state'
  # alpha = learning rate [0,1]
  # gamma = future discounting factor [0,1]
  #
  # Outputs:
  # Q = final learned Q-values
  # Q_evolution = every iteration of Q-values
  
  # Create output objects
  Q <- matrix(0, nrow = length(unique(data$state)), 
              ncol = length(unique(data$action)),
              dimnames = list(levels(data$state),
                              levels(data$action)))
  Q_evolution <- list()
  
  # Iterate over each trial in data
  for (i in seq_len(nrow(data))) {
    # Extract current values
    current_state  = data$state[i]
    current_action = data$action[i]
    next_state     = data$next_state[i]
    current_reward = data$reward[i]
    
    if (i == nrow(data)) {
      # Do nothing for last trial (which has no 'next state')
      Q[current_state, current_action] = Q[current_state, current_action]
      Q_evolution[[i]] <- data.frame(Q)
    } else {
      # Update via Q-learning rule
      Q[current_state, current_action] = Q[current_state, current_action] +
        alpha * (current_reward + gamma * max(Q[next_state, ]) - Q[current_state, current_action])
      Q_evolution[[i]] <- data.frame(Q)
    }
  }
  list(data.frame(Q), Q_evolution)
}
```

#### Learn
``` r
learned <- Q_learn(data, alpha = 0.08, gamma = 0.65)
```

#### Final Q-values
``` r
Q <- learned[[1]] 
```

#### Every iteration of Q-values
``` r
Q_evolution <- learned[[2]]  
```

#### Q-values approximate response proportions in data
``` r
prop.table(Q)
round(prop.table(table(data$state, data$action)), 2)
```

## Predict

#### Action prediction function
``` r
Q_predict <- function(data, Q) {
  # Create output objects
  action <- numeric(nrow(data))
  # Iterate over each trial in data
  for (i in seq_len(nrow(data))) {
    # Extract current state
    current_state = data$state[i]
    # Predict action (i.e., given a state, choose action with largest Q-value)
    action[i] = names(Q)[which(Q[current_state, ] == max(Q[current_state, ]))]
  }
  action
}
```

#### Get predictions
``` r
predicted <- Q_predict(data, Q)
```

#### Table predictions
``` r
table(data.frame(state = data$state, predicted))
round(prop.table(table(data.frame(state = data$state, predicted))), 2)
```

## Plot Q-value evolution

#### Some data wrangling to convert the `Q_evolution` list to a tidy data frame
``` r
df <- bind_rows(Q_evolution, .id = "trial")
df$trial <- as.numeric(df$trial)
df <- rownames_to_column(df, var = "state")
df$state[grepl("s1", df$state)] <- "s1"
df$state[grepl("s2", df$state)] <- "s2"
df$state[grepl("s3", df$state)] <- "s3"
df <- df %>% pivot_longer(cols = c("a1", "a2", "a3"), 
                          names_to = "action", 
                          values_to = "Q")
df$Q_normalized <- df$Q/max(df$Q)
head(df)
```

#### Plot
``` r
df %>% ggplot() +
  geom_line(aes(x = trial, 
                y = Q_normalized, 
                col = action)) +
  facet_grid(. ~ state) +
  labs(title = "Q-value evolution", 
       x = "Trial", 
       y = "Q", 
       col = "Action") +
  theme_minimal()
```
