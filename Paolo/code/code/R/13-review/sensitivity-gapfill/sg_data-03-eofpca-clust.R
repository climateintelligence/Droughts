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
# dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/data_long_HN_HS.rds")

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


# prcomp only full data ----------------------------------------------------------------

na_cols <- sapply(dat_hs_eof, function(x) any(is.na(x)))
mat_eof_sub <- mat_eof[, !na_cols]

pca_prcomp <- prcomp(mat_eof_sub, center = T, scale. = T, rank. = 20, retx = T)
pca_prcomp_summary <- summary_prcomp(pca_prcomp)


save(mat_eof_sub, pca_prcomp, pca_prcomp_summary,
     file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/sensitivity-gapfill/regions-02-pca-full.rda")


# kmeans on prcomp ---------------------------------------------------------
# 
# 
# mat_clust3 <- pca_prcomp$rotation[, 1:5]
# rownames(mat_clust3) <- colnames(mat_eof_sub)
# set.seed(1234)
# km_fit <- kmeans(mat_clust3, 5)
# stn_clust <- km_fit$cluster
# data.table(cluster = stn_clust, Name = names(stn_clust)) %>% 
#   merge(dat_meta, by = "Name") -> dat_plot_cluster
# 
# dat_plot_cluster[, cluster_fct := fct_recode(factor(cluster),
#                                              "South & high Alpine" = "1", 
#                                              "NE" = "2", 
#                                              "SE" = "3",
#                                              "NW" = "4", 
#                                              "North & high Alpine" = "5")]
# dat_plot_cluster[, cluster_fct := fct_relevel(cluster_fct, 
#                                               "NW", "NE", "North & high Alpine", 
#                                               "South & high Alpine", "SE")]
# 
# 
# saveRDS(dat_plot_cluster, 
#         "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/sensitivity-gapfill/meta-with-cluster-02-pca-full.rds")



