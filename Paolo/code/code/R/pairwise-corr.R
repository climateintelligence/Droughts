# look at pairwise correlations 
# by dist v & h


library(data.table)
library(magrittr)
library(ggplot2)
library(scico)
library(forcats)

mat_corr <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/SPATIAL_ANALYSES/ALL/correlation.rds")
mat_dist_h <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/SPATIAL_ANALYSES/ALL/distance.rds")
mat_dist_v <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/SPATIAL_ANALYSES/ALL/elevation_diff.rds")
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/SPATIAL_ANALYSES/ALL/list_of_sites.rds")



dat_corr <- data.table(corr = as.vector(mat_corr),
                       dist_h = as.vector(mat_dist_h),
                       dist_v = as.vector(mat_dist_v),
                       stn_x = rep(dat_meta$Name, times = nrow(dat_meta)),
                       stn_y = rep(dat_meta$Name, each = nrow(dat_meta)))

dat_corr <- dat_corr[!is.na(corr)]



# test plots --------------------------------------------------------------


summary(dat_corr)

breaks_h <- seq(0, 1000, by = 20) 
breaks_v <- seq(0, 3000, by = 50) 

dat_corr[, dist_h_cut := cut(dist_h, 
                             breaks = breaks_h, 
                             labels = breaks_h[-length(breaks_h)],
                             include.lowest = T)]
dat_corr[, dist_v_cut := cut(dist_v, 
                             breaks = breaks_v, 
                             labels = breaks_v[-length(breaks_v)],
                             include.lowest = T)]
dat_corr %>% str
dat_corr[, xx := as.numeric(as.character(dist_h_cut))]
dat_corr[, yy := as.numeric(as.character(dist_v_cut))]


dat_plot <- dat_corr[, .(corr_q05 = quantile(corr, 0.05),
                         corr_q50 = quantile(corr, 0.5),
                         corr_q95 = quantile(corr, 0.95),
                         corr_q90= quantile(corr, 0.9),
                         corr_mean = mean(corr)),
                     .(xx, yy)]

ggplot(dat_plot, aes(xx, yy, fill = corr_mean))+
  geom_raster()+
  theme_bw()+
  scale_fill_binned(type = "viridis")
  

dat_plot_m <- melt(dat_plot, id.vars = c("xx", "yy"), measure.vars = c("corr_q50", "corr_q90") )
dat_plot_m %>% summary

dat_plot_m[, variable_f := fct_recode(variable,
                                     "Median" = "corr_q50",
                                     "90th percentile" = "corr_q90")]
dat_plot_m[, value_f := cut(value, breaks = c(-0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1))]


ggplot(dat_plot_m, aes(xx, yy, fill = value_f))+
  geom_raster()+
  theme_bw()+
  # scale_fill_binned("Correlation",
  #                   type = "viridis", 
  #                   breaks = c(-0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
  #                   show.limits = T)+
  scale_fill_scico_d(direction = 1, palette = "roma")+
  facet_wrap(~variable_f)+
  xlim(0, 400)+ylim(0,1000)+
  xlab("Horizontal distance [km]")+
  ylab("Vertical distance [m]")+
  ggtitle("Summary of pairwise correlations between stations")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/SPATIAL_ANALYSES/ALL/distance_eleDiff_corr/all.png",
       width = 8, height = 4)



# div by altitude ---------------------------------------------------------

dat_corr %>% 
  merge(dat_meta[, .(stn_x = Name, elev_x = Elevation)], by = "stn_x") %>% 
  merge(dat_meta[, .(stn_y = Name, elev_y = Elevation)], by = "stn_y") -> dat_corr2

dat_corr2[, elev_grp := "mixed"]
dat_corr2[elev_x < 1000 & elev_y < 1000, elev_grp := "below 1000m"]
dat_corr2[elev_x >= 1000 & elev_y >= 1000, elev_grp := "above 1000m"]

# dat_corr2[, elev_grp := ifelse(elev_x < 1000 & elev_)]
# dat_corr2[elev_x < 1000 & elev_y < 1000] %>% summary
# dat_corr2[!(elev_x < 1000 & elev_y < 1000)] %>% summary



dat_plot2 <- dat_corr2[, .(corr_q05 = quantile(corr, 0.05),
                           corr_q50 = quantile(corr, 0.5),
                           corr_q95 = quantile(corr, 0.95),
                           corr_q90= quantile(corr, 0.9),
                           corr_mean = mean(corr)),
                       .(elev_grp, xx, yy)]


dat_plot_m2 <- melt(dat_plot2, id.vars = c("elev_grp", "xx", "yy"),
                    measure.vars = c("corr_q50", "corr_q90") )
dat_plot_m2 %>% summary

dat_plot_m2[, variable_f := fct_recode(variable,
                                     "Median" = "corr_q50",
                                     "90th percentile" = "corr_q90")]
dat_plot_m2[, value_f := cut(value, breaks = c(-0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1))]

ggplot(dat_plot_m2, aes(xx, yy, fill = value_f))+
  geom_raster()+
  theme_bw()+
  # scale_fill_binned("Correlation",
  #                   type = "viridis", 
  #                   breaks = c(-0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
  #                   show.limits = T)+
  scale_fill_scico_d(direction = 1, palette = "roma")+
  facet_grid(variable_f ~ elev_grp)+
  xlim(0, 400)+ylim(0,1000)+
  xlab("Horizontal distance [km]")+
  ylab("Vertical distance [m]")+
  ggtitle("Summary of pairwise correlations between stations")

ggsave(filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/SPATIAL_ANALYSES/ALL/distance_eleDiff_corr/elev-group.png",
       width = 8, height = 4)
