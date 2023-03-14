
# trends ------------------------------------------------------------------


dt_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
dt_lm <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/02_may_meeting/r-data/trends.rds")
dt_lm[, month_f := factor(month.abb[month], levels = month.abb[c(11,12,1:5)])]
dt_lm[, country := substr(Provider, 1, 2)]
dt_lm2 <- merge(dt_lm, 
                dt_meta,
                by = c("Provider", "Name"))

dt_year0 <- dt_lm2[term == "year0"]
# remove zugspitze because on glacier
dt_year0 <- dt_year0[Name != "Zugspitze"]
# remove all 0
dt_year0_nonan <- dt_year0[!is.na(statistic)]

# first numbers -----------------------------------------------------------

dt_year0[qprob == 0.5, table(mw_start_year, month_f, snow)]
dt_year0[mw_start_year == 1980 & is.na(statistic), table(month_f, qprob, snow)]


# HS nov - may ---------------------------------------------------------------

dt_year0_nonan[mw_start_year == 1980 & qprob == "mean_hs" & snow == "HS"] %>% 
  ggplot(aes(Elevation, estimate))+
  geom_hline(yintercept = 0)+
  geom_point(size = 0.2)+
  # geom_point(size = 0.5, aes(colour = p.value < 0.1))+
  # geom_smooth(se = F)+
  # scale_colour_manual(values = c("grey", "black"))+
  coord_cartesian(ylim = c(-5, 5))+
  # facet_grid(country ~ month_f)+
  facet_wrap(~month_f, nrow = 2)+
  theme_bw()+
  xlab("Elevation [m]")+
  ylab("Trend mean monthly HS [cm/year]")+
  ggtitle("Linear trend (1980-1999) mean monthly HS")

ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/02_may_meeting/fig/trends-month.png",
       width = 8, height = 4)





# HS: moving window years -----------------------------------------------------

dt_year0_nonan[, mw_start_year_f := paste0(mw_start_year, "-", mw_start_year + 19)]

dt_year0_nonan[qprob == "mean_hs" & 
                 mw_start_year %in% c(1975, 1980, 1985, 1990, 1995) & 
                 month == 1] %>% 
  ggplot(aes(Elevation, estimate))+
  geom_hline(yintercept = 0)+
  geom_point(size = 0.2)+
  # geom_smooth(se = F, method = "gam", formula = y ~ s(x))+
  facet_wrap( ~ mw_start_year_f)+
  theme_bw()+
  xlab("Elevation [m]")+
  ylab("Trend mean monthly HS [cm/year]")+
  ggtitle("Linear trend January HS")




ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/02_may_meeting/fig/trends-mw.png",
       width = 8, height = 4)




# HS: qXX -----------------------------------------------------------------

gg1 <- dt_year0_nonan[mw_start_year == 1980 & snow == "HS" & qprob %in% c(0.1, 0.5, 0.9) &
                 month %in% c(12, 1:4)] %>% 
  dcast(Elevation + Name + month_f ~ paste0("q", qprob), value.var = "estimate") %>% 
  ggplot(aes(q0.5, q0.9))+
  # geom_point()+
  geom_bin2d(bins = 50)+
  scale_fill_viridis_c(trans = "log10")+
  geom_abline()+
  facet_grid(. ~ month_f)+
  theme_bw()+
  xlab("Trend median monthly HS [cm/year]")+
  ylab("Trend 90th percentile \n of monthly HS [cm/year]")

gg2 <- dt_year0_nonan[mw_start_year == 1980 & snow == "HS" & qprob %in% c(0.1, 0.5, 0.9) &
                        month %in% c(12, 1:4)] %>% 
  dcast(Elevation + Name + month_f ~ paste0("q", qprob), value.var = "estimate") %>% 
  ggplot(aes(q0.5, q0.1))+
  # geom_point()+
  geom_bin2d(bins = 50)+
  scale_fill_viridis_c(trans = "log10")+
  geom_abline()+
  facet_grid(. ~ month_f)+
  theme_bw()+
  xlab("Trend median monthly HS [cm/year]")+
  ylab("Trend 10th percentile \n of monthly HS [cm/year]")

gg_out <- gg1 / gg2




ggsave(gg_out,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/02_may_meeting/fig/trends-qprob.png",
       width = 12, height = 6)







# hs & hn -----------------------------------------------------------------

xylim <- dt_year0_nonan[mw_start_year == 1980 & qprob %in% c("mean_hs", "sum_hn"),
               range(estimate)]

dt_year0_nonan[mw_start_year == 1980 & qprob %in% c("mean_hs", "sum_hn")] %>% 
  dcast(Provider + Name + month_f + country + 
          Latitude + Longitude + Elevation ~ snow, value.var = "estimate") %>% 
  ggplot(aes(HS, HN))+
  geom_hline(yintercept = 0, linetype = "dashed")+
  geom_vline(xintercept = 0, linetype = "dashed")+
  geom_point(size = 0.2)+
  facet_wrap(~month_f, nrow = 2)+
  xlim(xylim)+ylim(xylim)+
  theme_bw()+
  ylab("Trend sum monthly HN [cm/year]")+
  xlab("Trend mean monthly HS [cm/year]")+
  ggtitle("Linear trend (1980-1999)")




ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/02_may_meeting/fig/trends-hs-vs-hn.png",
       width = 8, height = 4)
