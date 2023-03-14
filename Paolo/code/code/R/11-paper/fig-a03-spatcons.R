# figure for spatial consistency check



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(patchwork)
library(directlabels)
library(readxl)
library(writexl)
library(fs)

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/meta_wide_HS.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/data_long_HS.rds")
dat_hs2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")
dat_hs <- mitmatmisc::add_month_fct(dat_hs, 10)




# determine stns to check -------------------------------------------------

dat_sim <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/SC_5REF/MonthlySim_5Ref.rds")
setDT(dat_sim)

dat_obs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/data_wide_HS.rds")
dat_obs_sim <- merge(
  dat_obs %>% melt(id.vars = c("year", "month"), value.name = "obs", variable.name = "Name"),
  dat_sim %>% melt(id.vars = c("year", "month"), value.name = "sim", variable.name = "Name"),
  by = c("year", "month", "Name"),
  all = T
)

dat_djf <- dat_obs_sim[month %in% c(12, 1, 2), 
                       .(bias = mean(sim - obs, na.rm = T),
                         mae = mean(abs(sim - obs), na.rm = T),
                         rmse = sqrt(mean( (sim - obs)^2, na.rm = T )),
                         r2 = cor(sim, obs, use = "p")^2,
                         nn = sum(!is.na(obs) & !is.na(sim))),
                       .(Name)]

dat_metrics_djf <- merge(dat_djf, dat_meta, by = "Name")



dat_metrics_djf[, Ele_fct := cut(Elevation, 
                                 breaks = c(0, 250, 500, 750, 1000, 1250, 1500, 2000, 3000),
                                 dig.lab = 5)]

dat_metrics_djf_summ <- dat_metrics_djf[!is.na(bias), 
                                        .(bias_mu = mean(bias),
                                          bias_sd = sd(bias),
                                          bias_med = median(bias),
                                          bias_q25 = quantile(bias, 0.25),
                                          bias_q75 = quantile(bias, 0.75),
                                          bias_iqr = IQR(bias)),
                                        .(Ele_fct)]

dat_metrics_djf2 <- merge(dat_metrics_djf, dat_metrics_djf_summ, by = "Ele_fct")


dat_metrics_djf2[, in_sd := between(bias,
                                    bias_mu - 1.96 * bias_sd,
                                    bias_mu + 1.96 * bias_sd,)]

dat_metrics_djf2[, in_ab_mae := Elevation/1000*13 > mae]

stns_to_check <- dat_metrics_djf2[in_sd == F | r2 < 0.5 | in_ab_mae == F | is.na(bias),
                                  Name]





# plot metrics ------------------------------------------------------------


# bias
gg1 <- dat_metrics_djf2 %>% 
  ggplot(aes(bias, Elevation, colour = in_sd))+
  geom_point(na.rm = T, size = 0.7)+
  scale_color_grey()+
  # scale_color_manual(values = c("black", "grey70"))+
  theme_bw()+
  theme(legend.position = "blank")+
  ylab("Elevation [m]")+
  xlab("Bias [cm]")


# mae 
gg2 <- 
  dat_metrics_djf2 %>% 
  ggplot(aes(mae, Elevation, colour = in_ab_mae))+
  geom_point(na.rm = T, size = 0.7)+
  # geom_abline(slope = 13/1000, intercept = 0)+
  scale_color_grey()+
  theme_bw()+
  theme(legend.position = "blank")+
  ylab("Elevation [m]")+
  xlab("MAE [cm]")


# r2
gg3 <- 
  dat_metrics_djf2 %>% 
  ggplot(aes(r2, Elevation, colour = r2 > 0.5))+
  geom_point(na.rm = T, size = 0.7)+
  scale_color_grey()+
  theme_bw()+
  theme(legend.position = "blank")+
  ylab("Elevation [m]")+
  xlab("R squared")


gg_all <- wrap_plots(gg1, gg2, gg3)+
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")")


ggsave(gg_all,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/Figure A3.png",
       width = 12, height = 4)


# values ------------------------------------------------------------------


dat_metrics_djf2 %>% summary


