# check spatial consistency data

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/meta_wide_HS.rds")

dat_sim <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/SC_10REF/MonthlySim.rds")
setDT(dat_sim)

dat_metrics <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/SC_10REF/ErrorSpatCons_withEle.csv")

summary(dat_metrics)
dat_metrics[is.na(bias)]
dat_metrics$bias %>% qplot
dat_metrics$mae %>% qplot
dat_metrics$r2 %>% qplot

dat_metrics %>% 
  ggplot(aes(Ele, bias))+
  geom_point()+
  theme_bw()


dat_metrics %>% 
  ggplot(aes(Ele, mae))+
  geom_point()+
  theme_bw()


dat_metrics %>% 
  ggplot(aes(Ele, r2))+
  geom_point()+
  theme_bw()


dat_metrics %>% 
  ggplot(aes(Ele, bias))+
  geom_point()+
  facet_wrap(~Provider)+
  theme_bw()


dat_metrics %>% 
ggplot(aes(Ele, bias, colour = log(mae)))+
  geom_point()+
  theme_bw()+
  scale_colour_scico()


dat_metrics %>% 
  ggplot(aes(Ele, bias, colour = r2))+
  geom_point()+
  theme_bw()+
  scale_colour_binned(type = "viridis")


dat_metrics %>% 
  ggplot(aes(bias, r2, colour = Ele))+
  geom_point()+
  theme_bw()+
  scale_colour_binned(type = "viridis", n.breaks = 6)



# by elev class -----------------------------------------------------------

dat_metrics[, Ele_fct := cut(Ele, 
                             breaks = c(0, 250, 500, 750, 1000, 1250, 1500, 2000, 3000),
                             dig.lab = 5)]

dat_metrics[, Ele_fct := cut(Ele, 
                             breaks = c(0, 500, 1000, 1500, 2000, 3000),
                             dig.lab = 5)]

dat_metrics[, Ele_fct := cut(Ele, 
                             breaks = c(0, 1000, 2000, 3000),
                             dig.lab = 5)]

dat_metrics_summ <- dat_metrics[!is.na(bias), 
                                .(bias_mu = mean(bias),
                                  bias_sd = sd(bias),
                                  bias_med = median(bias),
                                  bias_q25 = quantile(bias, 0.25),
                                  bias_q75 = quantile(bias, 0.75),
                                  bias_iqr = IQR(bias)),
                                .(Ele_fct)]

dat_metrics_summ[, c("xmin", "xmax") := tstrsplit(Ele_fct, "[^0-9.-]+", keep = 2:3, type.convert = T)]

dat_metrics2 <- merge(dat_metrics, dat_metrics_summ)

dat_metrics2[, in_iqr := between(bias,
                                 bias_q25 - 1.5 * bias_iqr,
                                 bias_q75 + 1.5 * bias_iqr)]


dat_metrics2[, in_sd := between(bias,
                                bias_mu - 1.96 * bias_sd,
                                bias_mu + 1.96 * bias_sd,)]

dat_metrics2 %>% summary

dat_metrics2 %>% 
  ggplot(aes(Ele_fct, bias))+
  geom_boxplot()+
  geom_jitter(data = dat_metrics2[in_iqr == FALSE], color = "red", height = 0)+
  theme_bw()

dat_metrics %>% 
  ggplot(aes(bias))+
  geom_histogram()+
  theme_bw()+
  facet_wrap(~Ele_fct, scales = "free")

dat_metrics2 %>% 
  ggplot(aes(Ele, bias, colour = in_iqr))+
  geom_point()+
  theme_bw()

dat_metrics2 %>% 
  ggplot(aes(Ele, bias, colour = in_sd))+
  geom_point()+
  theme_bw()

dat_metrics2 %>% 
  ggplot(aes(Ele, bias))+
  geom_point()+
  theme_bw()

dat_metrics2 %>% 
  ggplot(aes(Ele, r2))+
  geom_point()+
  theme_bw()

dat_metrics2 %>% 
  ggplot(aes(bias, fill = in_iqr))+
  geom_histogram()+
  theme_bw()+
  facet_wrap(~Ele_fct, scales = "free")


dat_metrics2 %>% 
  ggplot(aes(bias, fill = in_sd))+
  geom_histogram()+
  theme_bw()+
  facet_wrap(~Ele_fct, scales = "free")

dat_metrics2 %>% 
  ggplot(aes(bias, fill = r2 < 0.5))+
  geom_histogram()+
  theme_bw()+
  facet_wrap(~Ele_fct, scales = "free")

dat_metrics2[in_sd == F] %>% summary
dat_metrics2[r2 < 0.5]


dat_metrics2[in_sd == F | r2 < 0.5 | is.na(bias)]




# only dec-feb ------------------------------------------------------------

dat_obs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/07_SPATIAL_CONSISTENCY/data-to-check/data_wide_HS.rds")
dat_obs_sim <- merge(
  dat_obs %>% melt(id.vars = c("year", "month"), value.name = "obs", variable.name = "Name"),
  dat_sim %>% melt(id.vars = c("year", "month"), value.name = "sim", variable.name = "Name"),
  by = c("year", "month", "Name")
)

dat_djf <- dat_obs_sim[month %in% c(12, 1, 2), 
                       .(bias = mean(sim - obs, na.rm = T),
                         r2 = cor(sim, obs, use = "p")^2,
                         nn = sum(!is.na(obs) & !is.na(sim))),
                       .(Name)]
dat_djf %>% summary

dat_djf %>% merge(dat_meta, by = "Name") -> dat_djf_plot


dat_djf_plot %>% 
  ggplot(aes(Elevation, bias))+
  geom_point()+
  theme_bw()

dat_djf_plot %>% 
  ggplot(aes(Elevation, r2))+
  geom_point()+
  theme_bw()

dat_metrics_djf <- copy(dat_djf_plot)



# by elev class -----------------------------------------------------------

dat_metrics_djf[, Ele_fct := cut(Elevation, 
                                 breaks = c(0, 250, 500, 750, 1000, 1250, 1500, 2000, 3000),
                                 dig.lab = 5)]
# 
# dat_metrics_djf[, Ele_fct := cut(Ele, 
#                                  breaks = c(0, 500, 1000, 1500, 2000, 3000),
#                                  dig.lab = 5)]
# 
# dat_metrics_djf[, Ele_fct := cut(Ele, 
#                                  breaks = c(0, 1000, 2000, 3000),
#                                  dig.lab = 5)]

dat_metrics_djf_summ <- dat_metrics_djf[!is.na(bias), 
                                        .(bias_mu = mean(bias),
                                          bias_sd = sd(bias),
                                          bias_med = median(bias),
                                          bias_q25 = quantile(bias, 0.25),
                                          bias_q75 = quantile(bias, 0.75),
                                          bias_iqr = IQR(bias)),
                                        .(Ele_fct)]

dat_metrics_djf_summ[, c("xmin", "xmax") := tstrsplit(Ele_fct, "[^0-9.-]+", keep = 2:3, type.convert = T)]

dat_metrics_djf2 <- merge(dat_metrics_djf, dat_metrics_djf_summ, by = "Ele_fct")

dat_metrics_djf2[, in_iqr := between(bias,
                                     bias_q25 - 1.5 * bias_iqr,
                                     bias_q75 + 1.5 * bias_iqr)]


dat_metrics_djf2[, in_sd := between(bias,
                                    bias_mu - 1.96 * bias_sd,
                                    bias_mu + 1.96 * bias_sd,)]

dat_metrics_djf2 %>% summary


dat_metrics_djf2 %>% 
  ggplot(aes(Elevation, bias, colour = in_sd))+
  geom_point()+
  theme_bw()

dat_metrics_djf2 %>% 
  ggplot(aes(Elevation, bias))+
  geom_point()+
  theme_bw()

dat_metrics_djf2 %>% 
  ggplot(aes(Elevation, r2))+
  geom_point()+
  theme_bw()

dat_metrics_djf2 %>% 
  ggplot(aes(bias, fill = in_sd))+
  geom_histogram()+
  theme_bw()+
  facet_wrap(~Ele_fct, scales = "free")

dat_metrics_djf2 %>% 
  ggplot(aes(bias, fill = r2 < 0.5))+
  geom_histogram()+
  theme_bw()+
  facet_wrap(~Ele_fct, scales = "free")

dat_metrics_djf2[in_sd == F] %>% summary
dat_metrics_djf2[r2 < 0.5]


dat_metrics_djf2[in_sd == F | r2 < 0.5 | is.na(bias)]



