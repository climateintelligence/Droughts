# prep figure on longterm HS/HN changes

library(data.table)
library(magrittr)
library(ggplot2)
library(forcats)
library(slider)
#
dat <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/06_SEASONAL/Oct-Sep/data_long_HN_HS.rds")
meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/06_SEASONAL/Oct-Sep/meta_long_HN_HS.rds")



# subset data -------------------------------------------------------------

# HS
dat[!is.na(HS), .N, Name] %>% summary

dat_hs_overview <- dat[!is.na(HS), 
                       .(nn = .N, 
                         year_start = min(hydro_year), 
                         year_end = max(hydro_year),
                         max_years = max(hydro_year) - min(hydro_year) + 1), 
                       Name]
ggplot(dat_hs_overview, aes(year_start, year_end, fill = nn))+
  geom_tile()+
  scale_fill_viridis_c()

dat_hs_overview %>% 
  merge(meta) %>% 
  ggplot(aes(year_start, year_end))+
  geom_tile()+
  facet_wrap(~Provider)+
  theme_bw()

dat_hs_overview[year_start < 1920]
stn_sub_hs <- dat_hs_overview[year_start < 1920, Name]


dat_hs_overview[year_start < 1930 & year_end > 2000 & nn >= (max_years - 5)]
stn_sub_hs <- dat_hs_overview[year_start < 1930 & year_end > 2000 & nn >= (max_years - 5), Name]


# HN
dat[!is.na(HN), .N, Name] %>% summary

dat_hn_overview <- dat[!is.na(HN), 
                       .(nn = .N, 
                         year_start = min(hydro_year), 
                         year_end = max(hydro_year),
                         max_years = max(hydro_year) - min(hydro_year) + 1), 
                       Name]
dat_hn_overview <- merge(dat_hn_overview, meta[, .(Name, Provider)])
ggplot(dat_hn_overview, aes(year_start, year_end, fill = nn))+
  geom_tile()+
  scale_fill_viridis_c()

dat_hn_overview[year_start < 1900 & nn > 80 & year_end > 2000]
stn_sub_hn <- dat_hn_overview[year_start < 1900 & nn > 80 & year_end > 2000, Name]


dat_hn_overview[year_start < 1900 & year_end > 2000 & nn >= (max_years - 5)]
stn_sub_hn <- dat_hn_overview[year_start < 1900 & year_end > 2000 & nn >= (max_years - 5), Name]


# prep sub data -----------------------------------------------------------


dat_plot_hs <- dat[Name %in% stn_sub_hs]
dat_meta_hs <- meta[Name %in% stn_sub_hs]
dat_meta_hs$Elevation %>% qplot
dat_meta_hs$Elevation %>% summary
dat_meta_hs[, fct_elev := cut(Elevation, breaks = c(0, 500, 1000, 1500), dig.lab = 5)]

dat_plot_hs <- merge(dat_plot_hs, dat_meta_hs)

setkey(dat_plot_hs, Name, hydro_year)
# dat_plot_hs[, .(hydro_year = seq(min(hydro_year), max(hydro_year))), Name]
dat_plot_hs[, 
            HS_window := slide_dbl(HS, mean, na.rm = T, 
                                   .before = 15, .after = 15, .complete = F),
            Name]

dat_plot_hn <- dat[Name %in% stn_sub_hn]
dat_meta_hn <- meta[Name %in% stn_sub_hn]
dat_meta_hn$Elevation %>% qplot
dat_meta_hn$Elevation %>% summary
dat_meta_hn[, fct_elev := cut(Elevation, breaks = c(0, 500, 1000, 1500, 2500), dig.lab = 5)]

dat_plot_hn <- merge(dat_plot_hn, dat_meta_hn)

setkey(dat_plot_hn, Name, hydro_year)
# dat_plot_hn[, .(hydro_year = seq(min(hydro_year), max(hydro_year))), Name]
dat_plot_hn[, 
            HN_window := slide_dbl(HN, mean, na.rm = T, 
                                   .before = 15, .after = 15, .complete = F),
            Name]

# try plots ---------------------------------------------------------------


dat_plot_hs %>% 
  ggplot(aes(hydro_year, HS, colour = fct_elev))+
  geom_line(aes(group = Name))+
  theme_bw()+
  facet_grid(fct_elev ~ .)


dat_plot_hn %>% 
  ggplot(aes(hydro_year, HN, colour = fct_elev))+
  geom_line(aes(group = Name))+
  theme_bw()+
  facet_grid(fct_elev ~ ., scales = "free_y")


dat_plot_hn[Name == "Torino"] %>% 
  ggplot(aes(hydro_year, HN, colour = fct_elev))+
  geom_line(aes(group = Name))+
  theme_bw()+
  facet_grid(fct_elev ~ ., scales = "free_y")

dat[Name %in% c("Torino", "Riva_del_Garda", "Trento_Laste",
                "Osservatorio_Meteorologico_Storico_di_San_Rocco")] %>% 
  ggplot(aes(hydro_year, HN, colour = Name))+
  geom_line(aes(group = Name))+
  theme_bw()+
  facet_wrap(~Name, ncol = 1)

dat_plot_hn %>% 
  ggplot(aes(hydro_year, HN, colour = Name))+
  geom_line(aes(group = Name))+
  theme_bw()+
  facet_wrap(~ fct_elev, scales = "free_y", ncol = 1)



dat_plot_hn %>% 
  ggplot(aes(hydro_year, HN))+
  geom_line(aes(group = Name), colour = "grey30")+
  # geom_line(aes(y = HN_window), colour = "blue", size = 1)+
  theme_bw()+
  facet_wrap(~ fct_reorder(Name, Elevation), scales = "free_y")+
  ylim(0, NA)

# this is ok?



# final plots? ------------------------------------------------------------


dat_plot_hn %>% 
  ggplot(aes(hydro_year, HN))+
  geom_line(aes(group = Name), colour = "grey30")+
  # geom_line(aes(y = HN_window), colour = "blue", size = 1)+
  theme_bw()+
  facet_wrap(~ fct_reorder(Name, Elevation), scales = "free_y")+
  ylim(0, NA)


dat_plot_hs %>% 
  ggplot(aes(hydro_year, HS))+
  geom_line(aes(group = Name), colour = "grey30")+
  # geom_line(aes(y = HS_window), colour = "blue", size = 1)+
  theme_bw()+
  facet_wrap(~ fct_reorder(Name, Elevation), scales = "free_y")+
  ylim(0, NA)

# HS not so obvious, maybe leave it to HN?



# for snow dossier? -------------------------------------------------------

# dat[Name %in% c("Torino", "Riva_del_Garda", "Trento_Laste",
#                 "Osservatorio_Meteorologico_Storico_di_San_Rocco")] %>% 
dat[Name %in% c("Torino", 
                "Osservatorio_Meteorologico_Storico_di_San_Rocco")] %>% 
  merge(meta) %>% 
  ggplot(aes(hydro_year, HN))+
  geom_line(aes(group = Name))+
  theme_bw()+
  facet_wrap(~ fct_reorder(Name, Elevation), ncol = 1)
