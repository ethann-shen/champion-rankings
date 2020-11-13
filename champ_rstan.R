
library(tidyverse)
library(rstan)

final_data <- readRDS(file = "final_data_sub2.Rds")
# final_data %>% select(-(largestKillingSpree:`damageTakenDiffPerMinDeltas 30-end`), -(X1:status.status_code)) %>% saveRDS("final_data_sub2.Rds")
win_perc_tbl <- final_data %>% 
  group_by(team_comp) %>% 
  mutate(win_count = sum(ifelse(win == "True", 1, 0)) / 5,
         comp_count = n()/5) %>% 
  mutate(win_perc = round(win_count / comp_count, digits = 4)) %>% 
  select(team_comp, win_perc, win_count, comp_count) %>% 
  distinct()

temp_bt_df <- final_data %>% 
  ungroup() %>% 
  mutate(d = row_number()) %>% 
  select(d, win, team_comp) %>% 
  mutate(win = case_when(
    win == "False" ~ "loser", 
    win == "True" ~ "winner"
  )) %>% 
  tidyr::spread(key=win, value = team_comp) 

predictors <- final_data %>%
  ungroup() %>%
  mutate(win = case_when(
    win == "False" ~ 0,
    win == "True" ~ 1
  ) %>% as.factor(),
  championPoints =  scale(championPoints),
  lastPlayTime = scale(lastPlayTime),
  totalChampion_mastery_score = scale(totalChampion_mastery_score),
  chestGranted = ifelse(chestGranted == TRUE, 1, 0),
  gameId = rownum) #%>%
  # group_by(rownum, teamId) %>%
  # mutate(total_team_kills = sum(kills, na.rm=TRUE),
  #        KP = ifelse(kills == 0 & assists == 0, 0, (kills + assists) / total_team_kills),
  #        KDA = ifelse(deaths == 0, kills + assists, (kills + assists) / deaths)) %>%
  # ungroup() %>%
  #mutate()

contest <- bind_cols(
  temp_bt_df %>% select(winner) %>% na.omit(),
  temp_bt_df %>% select(loser) %>% na.omit()
) %>% 
  slice(which(row_number() %% 5 == 1)) 

matches <- contest %>% 
  mutate(winner = factor(winner, 
                         levels = c(contest$winner, contest$loser) %>% unique() %>% sort()),
         loser = factor(loser, 
                        levels = c(contest$winner, contest$loser) %>% unique() %>% sort())
  ) 

BT_predictors <- predictors %>% 
  select(win, rownum, championPoints, totalChampion_mastery_score, lastPlayTime, team_comp) %>% 
  mutate(team_comp = factor(team_comp,
                            levels = c(as.character(matches$winner), as.character(matches$loser)) %>% unique() %>% sort()),
         championPoints = scale(championPoints) %>% as.numeric(), 
         totalChampion_mastery_score = scale(totalChampion_mastery_score) %>% as.numeric(), 
         lastPlayTime = scale(lastPlayTime) %>% as.numeric())

matches_cols <- readRDS("matches.Rds")
matches_fit <- matches_cols %>% 
  select(-winner_num, -loser_num) %>% 
  mutate(outcome = as.numeric(outcome)) %>% 
  bind_cols(
    BT_predictors %>% 
      group_by(rownum, win) %>% 
      summarise(sum_ChampPoints = sum(championPoints, na.rm = TRUE),
                sum_totalChampion_mastery_score = sum(totalChampion_mastery_score, na.rm = TRUE),
                sum_lastPlayTime = sum(lastPlayTime, na.rm = TRUE)) %>% 
      ungroup() %>% 
      group_by(rownum) %>% 
      mutate(diff_ChampPoints = sum_ChampPoints - lag(sum_ChampPoints),
             diff_totalChampion_mastery_score = sum_totalChampion_mastery_score - lag(sum_totalChampion_mastery_score),
             diff_lastPlayTime = sum_lastPlayTime - lag(sum_lastPlayTime)) %>% 
      na.omit() %>% 
      ungroup() %>% 
      select(diff_ChampPoints:diff_lastPlayTime)
  )

hierarchical_model_vars = "
data {
  int<lower = 0> N_games;                    
  int<lower = 0> N_comps;  
  int<lower = 0> N_predictors;
  int<lower = 0, upper = 1> y[N_games]; 
  matrix[N_games, N_comps+N_predictors] X;

}
parameters {
 vector[N_comps+N_predictors] beta;
 real<lower = 0> sigma_comp;                 
}
// transformed parameters {
//   vector[N_games] y_pred = X * beta;
// }
model {
  for (i in 1:N_games) {
    y[i] ~ bernoulli_logit((X[i,] * beta));
  }

  for(i in 1:N_comps){
     beta[i] ~ normal(0,sigma_comp);
  }
  
  for (j in (N_comps + 1):(N_comps + N_predictors)) {
     beta[j] ~ normal(0,10);
  }

  sigma_comp ~ gamma(1,1); 
  
}

generated quantities {
  // int<lower=1, upper=N_comps+N_predictors> ranking[N_comps+N_predictors];       // rank of player ability
  vector[N_comps+N_predictors] log_lik;
  // {
  //   int ranked_index[N_comps+N_predictors] = sort_indices_asc(beta);
  //   for (k in 1:N_comps+N_predictors)
  //     ranking[ranked_index[k]] = k;
  // }
  
  for (n in 1:N_comps+N_predictors) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | X[n] * beta);
  }
}
"
set.seed(20200922)
bound <- floor((nrow(matches_fit)/4)*3)         #define % of training and test set

matches_fit <- matches_fit[sample(nrow(matches_fit)), ]           #sample rows 
matches_fit.train <- matches_fit[1:bound, ]              #get training set
matches_fit.test <- matches_fit[(bound+1):nrow(matches_fit), ]  
stan_data_vars <- list(
  y = matches_fit.test %>%  pull(outcome) %>% as.numeric(),
  N_comps = 382,
  N_predictors = 4,
  N_games = nrow(matches_fit.test), # matches
  X = matches_fit.test %>% mutate(int_eff = diff_ChampPoints*diff_totalChampion_mastery_score) %>% select(-outcome)  %>% as.matrix() 
)

rstan::stan(
  model_code = hierarchical_model_vars, 
  data = stan_data_vars,
  iter = 500,
  chains = 1
) %>% saveRDS("test_set.Rds")

readRDS("test_set.Rds")

