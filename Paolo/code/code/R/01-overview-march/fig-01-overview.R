# overview figures, tables and maps

library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(flextable)
library(officer)

# prep data ---------------------------------------------------------------



dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/meta-01-original.rds")
dat_hs <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hs-01-original.rds")
dat_hn <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hn-01-original.rds")

dat_hs[, month := month(date)]
dat_hn[, month := month(date)]

dat_hs[, season_year := year(date)]
dat_hn[, season_year := year(date)]

dat_hs[month <= 8, season_year := season_year - 1L]
dat_hn[month <= 8, season_year := season_year - 1L]

# info in --------------------------------------------------------------------



dat_meta
dat_hs_stn <- unique(dat_hs[, .(provider, stn_name)])
dat_hn_stn <- unique(dat_hn[, .(provider, stn_name)])

dat_meta[, .(provider, stn_name = Name, in_meta = "meta")] %>% 
  merge(dat_hs_stn[, .(provider, stn_name, in_hs = "hs")], all = T) %>% 
  merge(dat_hn_stn[, .(provider, stn_name, in_hn = "hn")], all = T) -> dat_in

dat_in[is.na(in_meta)]




# subset1 -----------------------------------------------------------------

# hs
dat_hs_avail <- unique(dat_hs[!is.na(hs), .(provider, stn_name, season_year)])
dat_hs_avail[, 
             ":="(ymin = min(season_year),
                    ymax = max(season_year),
                    n_years = length(unique(season_year))),
             .(provider, stn_name)]
dat_hs_avail[, subset1 := dplyr::if_else(ymin <= 1990 & ymax >= 2010 & n_years >= 15,
                                         "selected",
                                         "removed")]
dat_hs_avail_unique <- unique(dat_hs_avail[, .(provider, stn_name, subset1)])
stn_hs_subset1 <- dat_hs_avail_unique[subset1 == "selected", stn_name]

# hn
dat_hn_avail <- unique(dat_hn[!is.na(hn), .(provider, stn_name, season_year)])
dat_hn_avail[, 
             ":="(ymin = min(season_year),
                  ymax = max(season_year),
                  n_years = length(unique(season_year))),
             .(provider, stn_name)]
dat_hn_avail[, subset1 := dplyr::if_else(ymin <= 1990 & ymax >= 2010 & n_years >= 15,
                                         "selected",
                                         "removed")]
dat_hn_avail_unique <- unique(dat_hn_avail[, .(provider, stn_name, subset1)])
stn_hn_subset1 <- dat_hn_avail_unique[subset1 == "selected", stn_name]


# fig n stn by year -------------------------------------------------------

dat_plot <- rbindlist(list(hs = dat_hs_avail[subset1 == "selected"],
                           hn = dat_hn_avail[subset1 == "selected"]),
                      idcol = "snow")
dat_plot[, country := substr(provider, 1, 2)]

dat_plot[, 
         .(n_stn = .N),
         .(snow, country, season_year)] %>% 
ggplot(aes(season_year, n_stn, colour = snow))+
  geom_line()+
  facet_wrap(~country, scales = "free")+
  theme_bw()+
  ylim(0,NA)+
  xlab(NULL)+
  ylab("# stations per year")
  
ggsave(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/fig/number_station_by_year.png",
       width = 8, height = 4, units = "in")




# altitude distribution of 1990-2010 stations -----------------------------

rbindlist(list(hs = dat_hs_avail[subset1 == "selected"],
               hn = dat_hn_avail[subset1 == "selected"]),
          idcol = "snow") %>% 
  .[season_year >= 1990 & season_year <= 2010,
    .(snow, country = substr(provider, 1, 2), provider, stn_name)] %>% 
  unique %>% 
  merge(dat_meta[, .(provider, stn_name = Name, elev = Elevation)],
        by = c("provider", "stn_name")) -> dat_plot

dat_plot %>% 
  ggplot(aes(elev))+
  geom_histogram(binwidth = 100)+
  facet_wrap(~country, scales = "free_y")+
  theme_bw()+
  xlab("Station altitude [m]")+
  ylab("# stations")+
  ggtitle("Altitude distribution for 1990-2010 stations")

ggsave(file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/fig/altitude_stations_1990_2010.png",
       width = 8, height = 4, units = "in")



# number of available obs per month ---------------------------------------

# hs 

dat_hs_nobs <- dat_hs[!is.na(hs) & stn_name %in% stn_hs_subset1, 
                      .(n_obs = .N, min_date = min(date)),
                      .(provider, stn_name, season_year, month)]
dat_hs_nobs[, n_obs_max := days_in_month(min_date)]
dat_hs_nobs[, perc_obs := n_obs / n_obs_max]
dat_hs_nobs[, country := substr(provider, 1, 2)]
dat_hs_nobs[, ymin := min(season_year)]
dat_hs_nobs[, ymax := max(season_year)]

all_country <- sort(unique(dat_hs_nobs$country))

grp_size <- 50

pdf("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/fig/percent_available_observations_hs_detail.pdf",
    width = 20, height = 10)
for(i_country in all_country){
  
  dat_plot <- dat_hs_nobs[country == i_country]
  setkey(dat_plot, ymin, ymax, season_year)
  
  dat_plot[, stn_name_f := fct_inorder(stn_name)]
  dat_plot[, stn_name_n := as.numeric(stn_name_f)]
  dat_plot[, grp := floor(stn_name_n/grp_size)]
  
  grp_all <- sort(unique(dat_plot$grp))
  for(i_grp in grp_all){
    
    dat_plot2 <- dat_plot[grp == i_grp]
    dat_plot2[, date_x := floor_date(min_date, "month")]
    dat_plot2[, date_x_end := ceiling_date(min_date, "month")]
    dat_plot2[, month_f := factor(month.abb[month], levels = month.abb)]
    
    # dat_plot2 %>% 
    # ggplot(aes(date_x, fct_rev(stn_name_f), fill = perc_obs))+
    #   geom_tile(aes(width = n_obs_max))+
    #   theme_bw()+
    #   scale_fill_viridis_c(direction = -1)+
    #   xlab(NULL)+
    #   ylab(NULL)+
    #   ggtitle()
    
    gg <- dat_plot2 %>% 
      ggplot(aes(season_year, fct_rev(stn_name_f), fill = perc_obs))+
      geom_tile(width = 1)+
      theme_bw()+
      facet_grid(. ~ month_f)+
      scale_fill_viridis_c(direction = -1, limits = c(0,1))+
      xlim(1960, 2020)+
      xlab(NULL)+
      ylab(NULL)+
      ggtitle(paste0(i_country, ": ", i_grp + 1, " of ", length(grp_all)),
              "percent of available observations by month and year")
    
    print(gg)
    
  }
  
  
}
dev.off()



# hn 

dat_hn_nobs <- dat_hn[!is.na(hn) & stn_name %in% stn_hn_subset1, 
                      .(n_obs = .N, min_date = min(date)),
                      .(provider, stn_name, season_year, month)]
dat_hn_nobs[, n_obs_max := days_in_month(min_date)]
dat_hn_nobs[, perc_obs := n_obs / n_obs_max]
dat_hn_nobs[, country := substr(provider, 1, 2)]
dat_hn_nobs[, ymin := min(season_year)]
dat_hn_nobs[, ymax := max(season_year)]

all_country <- sort(unique(dat_hn_nobs$country))

grp_size <- 50

pdf("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/fig/percent_available_observations_hn_detail.pdf",
    width = 20, height = 10)
for(i_country in all_country){
  
  dat_plot <- dat_hn_nobs[country == i_country]
  setkey(dat_plot, ymin, ymax, season_year)
  
  dat_plot[, stn_name_f := fct_inorder(stn_name)]
  dat_plot[, stn_name_n := as.numeric(stn_name_f)]
  dat_plot[, grp := floor(stn_name_n/grp_size)]
  
  grp_all <- sort(unique(dat_plot$grp))
  for(i_grp in grp_all){
    
    dat_plot2 <- dat_plot[grp == i_grp]
    dat_plot2[, date_x := floor_date(min_date, "month")]
    dat_plot2[, date_x_end := ceiling_date(min_date, "month")]
    dat_plot2[, month_f := factor(month.abb[month], levels = month.abb)]
    
    # dat_plot2 %>% 
    # ggplot(aes(date_x, fct_rev(stn_name_f), fill = perc_obs))+
    #   geom_tile(aes(width = n_obs_max))+
    #   theme_bw()+
    #   scale_fill_viridis_c(direction = -1)+
    #   xlab(NULL)+
    #   ylab(NULL)+
    #   ggtitle()
    
    gg <- dat_plot2 %>% 
      ggplot(aes(season_year, fct_rev(stn_name_f), fill = perc_obs))+
      geom_tile(width = 1)+
      theme_bw()+
      facet_grid(. ~ month_f)+
      scale_fill_viridis_c(direction = -1, limits = c(0,1))+
      xlim(1960, 2020)+
      xlab(NULL)+
      ylab(NULL)+
      ggtitle(paste0(i_country, ": ", i_grp + 1, " of ", length(grp_all)),
              "percent of available observations by month and year")
    
    print(gg)
    
  }
  
  
}
dev.off()


# summary of available obs ------------------------------------------------

dat_hs_nobs[, month_f := factor(month.abb[month], levels = month.abb)]

dat_hs_nobs[season_year >= 1990 & season_year <= 2010] %>% 
  ggplot(aes(perc_obs*100))+
  geom_histogram(breaks = seq(0, 1, by = .2)*100)+
  facet_grid(country ~ month_f, scales = "free_y")+
  theme_bw()+
  scale_x_continuous(breaks = c(0.2, 0.4, 0.6, 0.8, 1)*100)+
  xlab("Available daily observations per month [%]")+
  ylab("# stations*years")+
  ggtitle("Distribution of available HS observation by month for all station-years 1990-2010")

ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/fig/percent_available_observations_hs_summary.png",
       width = 16, height = 8, units = "in")



dat_hn_nobs[, month_f := factor(month.abb[month], levels = month.abb)]

dat_hn_nobs[season_year >= 1990 & season_year <= 2010] %>% 
  ggplot(aes(perc_obs*100))+
  geom_histogram(breaks = seq(0, 1, by = .2)*100)+
  facet_grid(country ~ month_f, scales = "free_y")+
  theme_bw()+
  scale_x_continuous(breaks = c(0.2, 0.4, 0.6, 0.8, 1)*100)+
  xlab("Available daily observations per month [%]")+
  ylab("# stations*years")+
  ggtitle("Distribution of available HN observation by month for all station-years 1990-2010")

ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/fig/percent_available_observations_hn_summary.png",
       width = 16, height = 8, units = "in")

# tables ------------------------------------------------------------------

# hs hn both
dat_in[, table(in_hs, in_hn, useNA = "a")] %>% 
  data.frame %>% 
  .[-4, ] %>% 
  flextable %>% 
  autofit -> ft


# hs
dat_in[!(is.na(in_hs) & is.na(in_hn))] %>% 
  .[, table(in_hs, in_hn, useNA = "a")] %>%
  # addmargins() %>% 
  data.table %>% 
  # .[! (is.na(in_hs) & is.na(in_hn))] %>% 
  .[is.na(in_hs), in_hs := "no_hs"] %>% 
  .[is.na(in_hn), in_hn := "no_hn"] %>% 
  dcast(in_hs ~ in_hn, value.var = "N") %>% 
  # .[c(2,3,1)] %>% 
  flextable(col_keys = c("in_hs","hn", "no_hn")) %>% 
  set_header_labels(in_hs = "") %>% 
  autofit() -> ft

read_docx() %>% 
  body_add_flextable(ft) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/table/hs_hn.docx")


# hs
dat_hs_avail_unique[, table(provider, subset1)] %>%
  addmargins() %>% 
  data.table %>% 
  dcast(provider ~ subset1, value.var = "N") %>% 
  flextable(col_keys = c("provider","removed", "selected", "Sum")) %>% 
  autofit() -> ft

read_docx() %>% 
  body_add_flextable(ft) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/table/subset1_hs.docx")



# hn
dat_hn_avail_unique[, table(provider, subset1)] %>%
  addmargins() %>% 
  data.table %>% 
  dcast(provider ~ subset1, value.var = "N") %>% 
  flextable(col_keys = c("provider","removed", "selected", "Sum")) %>% 
  autofit() -> ft

read_docx() %>% 
  body_add_flextable(ft) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/table/subset1_hn.docx")



# maps --------------------------------------------------------------------

library(sf)
library(mapview)

dat_meta %>% 
  st_as_sf(coords = c("Longitude", "Latitude"),
           crs = sp::CRS("+proj=longlat +datum=WGS84")) %>% 
  mapview(zcol = "provider") #%>% 
  #mapshot(url = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/fig/map_all.html")



  dat_meta[Name %in% stn_hn_subset1 | Name %in% stn_hs_subset1] %>% 
  st_as_sf(coords = c("Longitude", "Latitude"),
           crs = sp::CRS("+proj=longlat +datum=WGS84")) %>% 
  mapview(zcol = "provider") #%>% 
 # mapshot(url = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/fig/map_subset1.html")

