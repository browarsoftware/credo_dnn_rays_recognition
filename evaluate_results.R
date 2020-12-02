my_path <- "d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_Kropki\\KropkiNaGithub\\" 
seed_val <- c(0, 101, 542, 1011, 3333, 4321, 6000, 7777, 10111, 15151)
mean_matrix <- matrix(rep(0, 4 *4), 4)
sd_matrix <- matrix(rep(0, 4 *4), 4)

network_name <- "VGG16"
#network_name <- "NASNetLarge"
#network_name <- "MobileNetV2"
#network_name <- "Xception"
#network_name <- "DenseNet201"


all_cf <- list()
for (a in 1:length(seed_val))
{
  my_path_help <- paste(my_path, "Results\\", network_name, seed_val[a],'_confusion_matrix_normalize.csv', sep='')
  cm <- read.csv(my_path_help, header = FALSE)
  all_cf[[a]] <- cm
}
for (a in 1:4)
{
  for (b in 1:4)
  {
    vec_help <- all_cf[[1]][a,b]
    for (c in 2:length(all_cf))
    {
      vec_help <- c(vec_help, all_cf[[c]][a,b])
    }
    #print(vec_help)
    mean_matrix[a,b] <- mean(vec_help)
    sd_matrix[a,b] <- sd(vec_help)
  }
}
mean_matrix <- mean_matrix * 100
sd_matrix <- sd_matrix * 100

dir.create(file.path(my_path, 'latex_array'), showWarnings = FALSE)

path_to_arrays <- paste(my_path,'\\latex_array\\', network_name, '.txt', sep = '')

sink(path_to_arrays)
for (a in 1:4)
{
  for (b in 1:4)
  {
    s <- sprintf("%.2f",mean_matrix[a,b])
    cat(s) 
    cat('$\\pm$')
    s <- sprintf("%.2f",sd_matrix[a,b])
    cat(s) 
    cat('&')
  }
  cat('\n')
}
sink()


all_cf <- list()
for (a in 1:length(seed_val))
{
  my_path_help <- paste(my_path, "Results\\", network_name, seed_val[a],'_confusion_matrix.csv', sep='')
  cm <- read.csv(my_path_help, header = FALSE)
  all_cf[[a]] <- cm
}
vec_help <- all_cf[[1]][1,1]
for (a in 2:4)
{
  vec_help <- vec_help + all_cf[[1]][a,a]
}
for (c in 2:length(all_cf))
{
  vec_help <- c(vec_help, all_cf[[c]][1,1])
  for (a in 2:4)
  {
    vec_help[c] <- vec_help[c] + all_cf[[c]][a,a]
  }
}
help_sum <- sum(all_cf[[1]])
vec_help <- vec_help / help_sum

sprintf("%.2f",mean(vec_help) * 100)
sprintf("%.2f",sd(vec_help) * 100)


my_path_help <- paste(my_path, "Results\\", network_name, seed_val[1],'_mymse.csv', sep='')
mse_vector = read.csv(my_path_help, header = FALSE)[1,1]
for (a in 2:length(seed_val))
{
  my_path_help <- paste(my_path, "Results\\", network_name, seed_val[a],'_mymse.csv', sep='')
  mse_vector <- c(mse_vector, read.csv(my_path_help, header = FALSE)[1,1])
}
  
sprintf("%.2f",mean(mse_vector))
sprintf("%.2f",sd(mse_vector))

#rysunki
random_vectors <- read.csv(paste(my_path, '\\RandomVectors.csv', sep = ''), header = FALSE)
random_vectors <- random_vectors + 1

dane <- read.csv(paste(my_path, '\\data\\data.txt', sep = ''))
actual_data <- dane[random_vectors$V1,2:5]

prediction <- read.csv(paste(my_path, '\\Results\\VGG160_2.csv', sep = ''), header = FALSE)

descriptions <- read.csv(paste(my_path, '\\data\\data.txt', sep = ''))
images_id <- descriptions$id[random_vectors$V1]

mean(rowMeans((actual_data - prediction)^2))

row_means_help <- rowMeans((actual_data - prediction)^2)
help_df <- data.frame(images_id, row_means_help, actual_data, prediction)
colnames(help_df) <- c('image','mse','A1','A1','A1','A4','P1','P2','P3','P4')
help_df_sorted <- help_df[order(help_df$mse),]



###########


colnames(actual_data) <- c('dots', 'lines', 'worms', 'artifacts')
colnames(prediction) <- c('dots', 'lines', 'worms', 'artifacts')

summary(sqrt(rowSums(dane[,2:5]^2)))

all_data <- rbind(actual_data, prediction)
group <- c(rep(1,nrow(actual_data)), rep(2,nrow(prediction)))

pairs(all_data, pch=c(1, 4)[group], col=c("red", "black")[group])


my_path <- "d:\\Projects\\Python\\PycharmProjects\\DLIB_Pytorch_Kropki\\KropkiNaGithub\\" 

seed_val <- c(0, 101, 542, 1011, 3333, 4321, 6000, 7777, 10111, 15151)

network_name <- "VGG16"
#network_name <- "NASNetLarge"
#network_name <- "MobileNetV2"
#network_name <- "Xception"
#network_name <- "DenseNet201"
all_cf <- list()
for (a in 1:length(seed_val))
{
  my_path_help <- paste(my_path, "checkpointsTest\\", network_name, seed_val[a],'.log', sep='')
  cm <- read.csv(my_path_help)
  all_cf[[a]] <- cm
}

plot(all_cf[[1]]$epoch, all_cf[[1]]$loss, type = "l", xlab = 'epoch', ylab='loss (MSE)', ylim = c(0, 0.12), col=1)
for (a in 2:length(all_cf))
{
  lines(all_cf[[a]]$epoch, all_cf[[a]]$loss, type = "l", xlab = 'epoch', ylab='loss (MSE)', ylim = c(0, 0.12), col=a)
}
legend(2900, 0.12, legend=c("Training set 1", "Training set 2", "Training set 3", "Training set 4", "Training set 5",
                            "Training set 6", "Training set 7", "Training set 8", "Training set 9","Training set 10"),
       col=1:10, lty=rep(1,10), cex=0.8)

###########

random_vectors <- read.csv(paste(my_path, '\\RandomVectors.csv', sep = ''), header = FALSE)
random_vectors <- random_vectors + 1
dane <- read.csv(paste(my_path, '\\data\\data.txt', sep = ''))
seed_val <- c(0, 101, 542, 1011, 3333, 4321, 6000, 7777, 10111, 15151)

table_list_t0 <- list()

for (seed_val_idx in 1:length(seed_val))
{
  actual_data <- dane[random_vectors[,seed_val_idx],2:5]
  
  
  prediction <- read.csv(paste(my_path,'\\Results\\VGG16', 
                               seed_val[seed_val_idx],'_2.csv', sep = ''), header = FALSE)
  
  
  help <- data.frame(prediction, actual_data)
  a <- 1
  pred_help <- help[a,1:4]
  pred_id <- which(max(pred_help) == pred_help)[1]
  actual_help <- help[a,1:4]
  actual_id <- which(max(actual_help) == actual_help)[1]
  for (a in 2:nrow(help))
  {
    pred_help <- help[a,1:4]
    pred_id <- c(pred_id, which(max(pred_help) == pred_help)[1])
    actual_help <- help[a,5:8]
    actual_id <- c(actual_id, which(max(actual_help) == actual_help)[1])
  }
  tab_1 <- table(pred_id, actual_id)
  table_list_t0[[seed_val_idx]] <- tab_1
}



tt <- 0.9
table_list <- list()
percent <- list()
eliminated_percent <- list()
recognition_rate <- list()

for (seed_val_idx in 1:length(seed_val))
{
  actual_data <- dane[random_vectors[,seed_val_idx],2:5]
  
  
  prediction <- read.csv(paste(my_path,'\\Results\\VGG16', 
                               seed_val[seed_val_idx],'_2.csv', sep = ''), header = FALSE)
  
  
  help <- data.frame(prediction, actual_data)
  idx <- help$V1 > tt
  idx <- idx | (help$V2 > tt)
  idx <- idx | (help$V3 > tt)
  idx <- idx | (help$V4 > tt)
  
  help <- help[idx,]
  percent[[seed_val_idx]] <- nrow(help)
  a <- 1
  pred_help <- help[a,1:4]
  pred_id <- which(max(pred_help) == pred_help)[1]
  actual_help <- help[a,1:4]
  actual_id <- which(max(actual_help) == actual_help)[1]
  for (a in 2:nrow(help))
  {
    pred_help <- help[a,1:4]
    pred_id <- c(pred_id, which(max(pred_help) == pred_help)[1])
    actual_help <- help[a,5:8]
    actual_id <- c(actual_id, which(max(actual_help) == actual_help)[1])
  }
  tab_1 <- table(pred_id, actual_id)
  
  recognition_rate[[seed_val_idx]] <- (tab_1[1,1] + tab_1[2,2] + tab_1[3,3] + tab_1[4,4]) / nrow(help)
  
  
  
  eliminated_percent[[seed_val_idx]] <- c(sum(tab_1[1,]) / sum(table_list_t0[[seed_val_idx]][1,]),
                                          sum(tab_1[2,]) / sum(table_list_t0[[seed_val_idx]][2,]),
                                          sum(tab_1[3,]) / sum(table_list_t0[[seed_val_idx]][3,]),
                                          sum(tab_1[4,]) / sum(table_list_t0[[seed_val_idx]][4,]))
    
  tab_1[1,] <- tab_1[1,] / sum(tab_1[1,])
  tab_1[2,] <- tab_1[2,] / sum(tab_1[2,])
  tab_1[3,] <- tab_1[3,] / sum(tab_1[3,])
  tab_1[4,] <- tab_1[4,] / sum(tab_1[4,])
  table_list[[seed_val_idx]] <- tab_1
}

mean_matrix <- matrix(rep(0, 4 *4), 4)
sd_matrix <- matrix(rep(0, 4 *4), 4)
for (a in 1:4)
{
  for (b in 1:4)
  {
    vec_help <- table_list[[1]][a,b]
    for (c in 2:length(table_list))
    {
      vec_help <- c(vec_help, table_list[[c]][a,b])
    }
    #print(vec_help)
    mean_matrix[a,b] <- mean(vec_help)
    sd_matrix[a,b] <- sd(vec_help)
  }
}
mean_matrix <- mean_matrix * 100
sd_matrix <- sd_matrix * 100
mean(unlist(percent)) / 235 * 100
sd(unlist(percent)) / 235 * 100

for (a in 1:4)
{
  for (b in 1:4)
  {
    s <- sprintf("%.2f",mean_matrix[a,b])
    cat(s) 
    cat('$\\pm$')
    s <- sprintf("%.2f",sd_matrix[a,b])
    cat(s) 
    cat('&')
  }
  cat('\n')
}


mean(unlist(recognition_rate)) * 100
sd(unlist(recognition_rate)) * 100

mat_help <- matrix(unlist(eliminated_percent), nrow = 10, byrow = TRUE) * 100
mean(mat_help[,1])
sd(mat_help[,1])
mean(mat_help[,2])
sd(mat_help[,2])
mean(mat_help[,3])
sd(mat_help[,3])
mean(mat_help[,4])
sd(mat_help[,4])

#####################
#Dataset statistic
dane <- read.csv(paste(my_path, '\\data\\data.txt', sep = ''))
actual_data <- dane[,2:5]
classes_count <- c(0,0,0,0)
artifacts <- 0
for (a in 1:nrow(actual_data))
{
  id_help <- which(max(actual_data[a,]) == actual_data[a,])[1]
  classes_count[id_help] <- classes_count[id_help] + 1
}

sum(classes_count)