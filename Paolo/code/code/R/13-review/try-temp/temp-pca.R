# check PCA eigenvectors, loadings, etc.


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)
library(scico)
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


dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_long_HN_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/gapfill-01-all.rds")

# subset to spatcons
stns_ok <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/01_submission/rds/spatcons_stns_ok.rds")
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



na_cols <- sapply(dat_hs_eof, function(x) any(is.na(x)))
mat_eof_sub <- mat_eof[, !na_cols]

pca_prcomp <- prcomp(mat_eof_sub, center = T, scale. = T, rank. = 20, retx = T)
pca_prcomp_summary <- summary_prcomp(pca_prcomp)



# plot --------------------------------------------------------------------



data.table(Name = colnames(mat_eof_sub), pca_prcomp$rotation) %>% 
  merge(dat_meta, by = "Name") -> dat_pca_prcomp

dat_pca_prcomp[, country := substr(Provider, 1, 2)]
# setnames(dat_pca_prcomp, paste0("V", 1:20), paste0("PC", 1:20))
dat_pca_prcomp %>% 
  melt(id.vars = c("country", "Provider", "Name", "Longitude", "Latitude", "Elevation"),
       measure.vars = paste0("PC", 1:5)) -> dat_pca_plot

# dat_pca_plot[, value_sc := scales::rescale(value, c(-1,1)), variable]
dat_pca_plot[, value_sc := value/sd(value), variable]
dat_pca_plot[, value_sc := value/sd(value), Name]
pca_prcomp_summary[, prop_sd_format := scales::percent(prop_sd, .1)]
pca_prcomp_summary[, facet_lbl := paste0(pc, " (", prop_sd_format, ")")]
pc_rename <- setNames(pca_prcomp_summary$facet_lbl, pca_prcomp_summary$pc)
dat_pca_plot[, facet_lbl := pc_rename[variable]]


dat_pca_plot %>% 
  ggplot(aes(Longitude, Latitude, colour = value_sc))+
  borders()+
  geom_point(size = 0.5)+
  scale_color_scico("Scaled loadings", palette = "roma",
                    guide = guide_colorbar(title.position = "top", direction = "horizontal"))+
  facet_wrap(~facet_lbl)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"), breaks = c(5,10,15))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()



# loadings as per stackoverflow -------------------------------------------

pca_prcomp$rotation %>% str

t(t(pca_prcomp$rotation) * pca_prcomp$sdev) %>% str

data.table(Name = colnames(mat_eof_sub), t(t(pca_prcomp$rotation) * pca_prcomp$sdev) ) %>% 
  merge(dat_meta, by = "Name") -> dat_pca_prcomp

dat_pca_prcomp[, country := substr(Provider, 1, 2)]
# setnames(dat_pca_prcomp, paste0("V", 1:20), paste0("PC", 1:20))
dat_pca_prcomp %>% 
  melt(id.vars = c("country", "Provider", "Name", "Longitude", "Latitude", "Elevation"),
       measure.vars = paste0("PC", 1:5)) -> dat_pca_plot

dat_pca_plot[, value_sc := value/sd(value), Name]
pca_prcomp_summary[, prop_sd_format := scales::percent(prop_sd, .1)]
pca_prcomp_summary[, facet_lbl := paste0(pc, " (", prop_sd_format, ")")]
pc_rename <- setNames(pca_prcomp_summary$facet_lbl, pca_prcomp_summary$pc)
dat_pca_plot[, facet_lbl := pc_rename[variable]]


dat_pca_plot %>% 
  ggplot(aes(Longitude, Latitude, colour = value_sc))+
  borders()+
  geom_point(size = 0.5)+
  scale_color_scico("Scaled loadings", palette = "roma",
                    guide = guide_colorbar(title.position = "top", direction = "horizontal"))+
  facet_wrap(~facet_lbl)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"), breaks = c(5,10,15))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_quickmap(xlim = range(dat_meta$Longitude), ylim = range(dat_meta$Latitude))+
  theme_bw()


