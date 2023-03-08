# visualize trends


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)


# prep data ---------------------------------------------------------------

dt_meta <- readRDS("data/meta-01-original.rds")
dt_lm <- readRDS("data/test-trend-01.rds")
dt_lm[, month_f := factor(month.abb[month], levels = month.abb[c(11,12,1:5)])]
dt_lm[, country := substr(provider, 1, 2)]
dt_lm2 <- merge(dt_lm, 
                dt_meta[, .(provider, stn_name = Name, 
                            lon = Longitude, lat = Latitude, elev = Elevation)])

dt_year0 <- dt_lm2[term == "year0"]
# remove zugspitze because on glacier
dt_year0 <- dt_year0[stn_name != "Zugspitze"]
# remove all 0
dt_year0_nonan <- dt_year0[!is.na(statistic)]

# first numbers -----------------------------------------------------------

dt_year0[qprob == 0.5, table(mw_start_year, month_f, snow)]
dt_year0[mw_start_year == 1980 & is.na(statistic), table(month_f, qprob, snow)]

# hs & hn -----------------------------------------------------------------

dt_year0_nonan[mw_start_year == 1980 & qprob %in% c("mean_hs", "sum_hn") & month == 1] %>% 
  ggplot(aes(elev, estimate))+
  geom_hline(yintercept = 0)+
  geom_point()+
  facet_grid(snow ~ country, scales = "free_y")+
  theme_bw()
# -> some stations missing

dt_year0_nonan[mw_start_year == 1980 & qprob %in% c("mean_hs", "sum_hn") & month == 1] %>% 
  dcast(provider + stn_name + month_f + country + lat + lon + elev ~ snow, value.var = "estimate") %>% 
  ggplot(aes(HN, HS))+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point()+
  facet_wrap(~country)+
  theme_bw()


# HS nov - may ---------------------------------------------------------------

dt_year0_nonan[mw_start_year == 1980 & qprob == 0.5 & snow == "HS"] %>% 
  ggplot(aes(month_f, estimate))+
  geom_boxplot()+
  coord_cartesian(ylim = c(-5, 2.5))

dt_year0_nonan[mw_start_year == 1980 & qprob == 0.5 & snow == "HS"] %>% 
  ggplot(aes(month_f, estimate))+
  geom_boxplot()+
  coord_cartesian(ylim = c(-5, 2.5))+
  facet_wrap(~country)

dt_year0_nonan[mw_start_year == 1980 & qprob == 0.5 & snow == "HS"] %>% 
  ggplot(aes(elev, estimate))+
  geom_hline(yintercept = 0)+
  geom_point()+
  geom_smooth(se = F)+
  # coord_cartesian(ylim = c(-5, 2.5))+
  facet_grid(country ~ month_f)+
  theme_bw()



# HS: moving window years -----------------------------------------------------

dt_year0_nonan[qprob == "mean_hs" & 
                 mw_start_year %in% c(1975, 1980, 1985, 1990, 1995) & 
                 month == 1] %>% 
  ggplot(aes(elev, estimate))+
  geom_hline(yintercept = 0)+
  geom_point()+
  geom_smooth(se = F, method = "gam", formula = y ~ s(x))+
  facet_grid(country ~ mw_start_year, scales = "free_y")+
  theme_bw()

dt_year0_nonan[qprob == "mean_hs" & 
                 mw_start_year %in% c(1975, 1980, 1985, 1990, 1995) &
                 month == 4] %>% 
  ggplot(aes(elev, estimate))+
  geom_hline(yintercept = 0)+
  geom_point()+
  geom_smooth(se = F, method = "gam", formula = y ~ s(x))+
  facet_grid(country ~ mw_start_year, scales = "free_y")+
  theme_bw()


# HS: qXX -----------------------------------------------------------------

dt_year0_nonan[mw_start_year == 1980 & snow == "HS" & month == 1 & qprob %in% c(0.1, 0.5, 0.9)] %>% 
  ggplot(aes(elev, estimate, colour = qprob))+
  geom_hline(yintercept = 0)+
  geom_point(size = 0.2)+
  geom_smooth(se = F, method = "gam", formula = y ~ s(x))+
  facet_wrap(~ country, scales = "free_y")+
  theme_bw()

dt_year0_nonan[mw_start_year == 1980 & snow == "HS" & qprob %in% c(0.1, 0.5, 0.9)] %>% 
  ggplot(aes(elev, estimate, colour = qprob))+
  geom_hline(yintercept = 0)+
  geom_point(size = 0.2)+
  geom_smooth(se = F, method = "gam", formula = y ~ s(x))+
  facet_grid(country ~ month_f, scales = "free_y")+
  theme_bw()


# latlon elev -------------------------------------------------------------

dt_plot <- dt_year0_nonan[mw_start_year == 1980 & qprob == "mean_hs" & month == 1] 
dt_plot[, estimate_abs := abs(estimate)]
dt_plot[, estimate_sign := sign(estimate)]


dt_plot %>% 
  ggplot(aes(lon, elev, colour = factor(estimate_sign), size = estimate_abs))+
  geom_point()+
  scale_size(range = c(0.1, 3))+
  facet_wrap(~country)+
  theme_bw()


dt_plot %>% 
  ggplot(aes(lat, elev, colour = factor(estimate_sign), size = estimate_abs))+
  geom_point()+
  scale_size(range = c(0.1, 3))+
  facet_wrap(~country)+
  theme_bw()
