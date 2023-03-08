# EOF analysis


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
# library(scico)
# library(lemon)
# library(patchwork)
# library(directlabels)
library(sinkr)

summary_sinkr_eof <- function(xx_eof, k = 10){
  sdev <- xx_eof$Lambda
  sdev <- sdev / sum(sdev)
  data.table(pc = paste0("PC", 1:k),
             prop_sd = sdev[1:k],
             cumsum_prop_sd = cumsum(sdev[1:k]))
}

summary_prcomp <- function(xx_pca, k = 10){
  sdev <- xx_pca$sdev^2
  sdev <- sdev / sum(sdev)
  data.table(pc = paste0("PC", 1:k),
             prop_sd = sdev[1:k],
             cumsum_prop_sd = cumsum(sdev[1:k]))
}

# prep data ---------------------------------------------------------------


dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")

# subset to spatcons
stns_ok <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/spatcons_stns_ok.rds")
dat_meta <- dat_meta[Name %in% stns_ok]
dat_hs <- dat_hs[Name %in% stns_ok]


setnames(dat_hs, "Date", "date")
dat_hs <- mitmatmisc::add_hydro_year(dat_hs)

dat_hs_sub <- dat_hs[hydro_year >= 1981 & hydro_year <= 2010 &
                       month(date) %in% c(12, 1:4)]

# subset based on # of available daily obs (better since based on daily data!)

dat_hs_sub[!is.na(HS), .N, .(Name)] %>% .[, qplot(N)]
dat_hs_sub[!is.na(HS), .N, .(Name)] %>% .[, max(N)] -> max_n
dat_hs_sub[!is.na(HS), .N, .(Name)] %>%
  .[N > 0.7*max_n] %>%
  merge(dat_hs_sub) -> dat_hs_sub2

# or subset based on gapfill data

# dat_meta_gp <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")
# dat_hs_sub2 <- dat_hs_sub[Name %in% dat_meta_gp$Name]

dat_hs_eof <- dcast(dat_hs_sub2, date ~ Name, value.var = "HS")


dat_hs_eof[, ":="(date = NULL)]
mat_eof <- as.matrix(dat_hs_eof)


# sinkr::eof all data ---------------------------------------------------------------------

sinkr_eof <- eof(mat_eof, centered = T, scaled = T, nu = 20, recursive = T)
sinkr_eof_summary <- summary_sinkr_eof(sinkr_eof)

save(mat_eof, sinkr_eof, sinkr_eof_summary,
     file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/regions-01-sinkr-eof.rda")

# kmeans on eof ---------------------------------------------------------


mat_clust3 <- sinkr_eof$u[, 1:5]
rownames(mat_clust3) <- colnames(mat_eof)
set.seed(1234)
km_fit <- kmeans(mat_clust3, 5)
stn_clust <- km_fit$cluster
data.table(cluster = stn_clust, Name = names(stn_clust)) %>% 
  merge(dat_meta, by = "Name") -> dat_plot_cluster

dat_plot_cluster[, cluster_fct := fct_recode(factor(cluster),
                                             "NW" = "1", 
                                             "NE" = "2",  
                                             "South & high Alpine" = "3",
                                             "SE" = "4", 
                                             "North & high Alpine" = "5")]
dat_plot_cluster[, cluster_fct := fct_relevel(cluster_fct, 
                                              "NW", "NE", "North & high Alpine", 
                                              "South & high Alpine", "SE")]


saveRDS(dat_plot_cluster, 
        "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/meta-with-cluster-01.rds")

saveRDS(km_fit, 
        "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/regions-03-km-obj.rds")



# prcomp only full data ----------------------------------------------------------------

na_cols <- sapply(dat_hs_eof, function(x) any(is.na(x)))
mat_eof_sub <- mat_eof[, !na_cols]

pca_prcomp <- prcomp(mat_eof_sub, center = T, scale. = T, rank. = 20, retx = T)
pca_prcomp_summary <- summary_prcomp(pca_prcomp)


save(mat_eof_sub, pca_prcomp, pca_prcomp_summary,
     file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/regions-02-pca-full.rda")


# kmeans on prcomp ---------------------------------------------------------


mat_clust3 <- pca_prcomp$rotation[, 1:5]
rownames(mat_clust3) <- colnames(mat_eof_sub)
set.seed(1234)
km_fit <- kmeans(mat_clust3, 5)
stn_clust <- km_fit$cluster
data.table(cluster = stn_clust, Name = names(stn_clust)) %>% 
  merge(dat_meta, by = "Name") -> dat_plot_cluster

dat_plot_cluster[, cluster_fct := fct_recode(factor(cluster),
                                             "South & high Alpine" = "1", 
                                             "NE" = "2", 
                                             "SE" = "3",
                                             "NW" = "4", 
                                             "North & high Alpine" = "5")]
dat_plot_cluster[, cluster_fct := fct_relevel(cluster_fct, 
                                              "NW", "NE", "North & high Alpine", 
                                              "South & high Alpine", "SE")]


saveRDS(dat_plot_cluster, 
        "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/meta-with-cluster-02-pca-full.rds")





# reconstruct HS based on eof ---------------------------------------------

n_pc <- 20

mat_test <- sinkr_eof$A[, 1:n_pc] %*% t(sinkr_eof$u[, 1:n_pc])
mat_test2 <- scale(mat_test, F, 1/sinkr_eof$F1_scale)
mat_test2 <- scale(mat_test2, -1 * sinkr_eof$F1_center, F)

# summary(as.vector(mat_eof - mat_test2))
# qplot(as.vector(mat_eof - mat_test2))

# i_sample <- sample(length(mat_eof), 1e4)
# i_sample_na <- !is.na(mat_eof[i_sample])
# plot(mat_eof[i_sample][i_sample_na], mat_test2[i_sample][i_sample_na])

# -> ok!

# sinkr_eof_summary
# summary_sinkr_eof(sinkr_eof, 20)

# mat_test2 %>% str

dat_hs_eof_4date <- dcast(dat_hs_sub2, date ~ Name, value.var = "HS")

dat_hs_rec_eof <- as.data.table(mat_test2)
setnames(dat_hs_rec_eof, colnames(mat_eof))
dat_hs_rec_eof[, date := dat_hs_eof_4date$date]

dat_hs_rec_eof2 <- melt(dat_hs_rec_eof, 
                        id.vars = "date",
                        variable.factor = F,
                        variable.name = "Name",
                        value.name = "HS_eof")

dat_hs_out <- merge(dat_hs_rec_eof2,
                    dat_hs_sub2[, .(Name, date, hydro_year, HS)],
                    by = c("Name", "date"))

saveRDS(dat_hs_out,
        file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/data-03-daily-from-eof.rds")
