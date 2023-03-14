


# FVG has hn and hs mixed -------------------------------------------------

fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/FVG/read_AINEVA_HS.csv") %>% 
  melt(id.vars = "Date", variable.factor = F) -> dat_fvg_hs
dat_fvg_hs[, Date := dmy(Date)]

dat_fvg_hs[startsWith(as.character(value), "9"), table(value)]
dat_fvg_hs[startsWith(as.character(value), "8"), table(value)]
dat_fvg_hs[value >= 800 & value < 900]
# dat_fvg[value >= 800 | value < 0, value := NA]



fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/FVG/read_AINEVA_HN.csv") %>% 
  melt(id.vars = "Date", variable.factor = F) -> dat_fvg
dat_fvg[, Date := dmy(Date)]
dat_fvg[startsWith(as.character(value), "9"), table(value)]
dat_fvg[startsWith(as.character(value), "8"), table(value)]
dat_fvg[value == 999 | value == 899, value := 0] # traces of HN or traces of HN with rain
dat_fvg[value == 800 , value := 0] # rain on snow, no fresh snow
dat_fvg[value > 800, value := value - 800]
dat_fvg[value < 0, value := NA]
dat_fvg[value > 100, .N, variable]
summary(dat_fvg)

dat_zz <- merge(dat_fvg_hs[, .(Date, variable, hs = value)],
                dat_fvg[, .(Date, variable, hn = value)], all = T)

zz <- dat_zz[variable == "Florianca"]
zz <- dat_zz[variable == "Casetta_in_Canada"]
zz <- dat_zz[variable == "Monte_Zoncolan"]
dat_fvg[value > 100, .N, variable]
dat_fvg[value > 100 & variable == "Florianca", .N, year(Date)]
dat_zz[hn > 100 & variable == "Casetta_in_Canada", .N, year(Date)]
dat_zz[hn > 100 & variable == "Monte_Zoncolan", .N, year(Date)]

dat_zz[!is.na(hs) & !is.na(hn), .(hs = sum(hs == 0), hn = sum(hn == 0)), variable]

all_stns <- sort(unique(dat_zz$variable))

pdf("fvg.pdf", width = 16, height = 8)
for(i_stn in all_stns){
  
  gg <- dat_zz[variable == i_stn & 
                 !is.na(hs) & !is.na(hn) & 
                 hs >= 0 & hn >= 0 & hs < 800 & hn < 800] %>% 
    ggplot(aes(Date))+
    geom_point(aes(y=hs, colour = "hs"))+
    geom_point(aes(y=hn, colour = "hn"))+
    facet_wrap(~year(Date), scales = "free_x")+
    theme_bw()+
    ggtitle(i_stn)
  print(gg)
}
dev.off()

all_stns

mixed <- c("Casetta_in_Canada", "Florianca", "Monte_Zoncolan", "Paularo","Pradibsco", "Varmost")


# test normalize ----------------------------------------------------------



# need to normalize data because of heteroscedascity 

dat_intercept <- dat_lm2[term == "(Intercept)" & estimate > 0] # remove the single negative value

dat_intercept[, estimate_sqrt := sqrt(estimate)]
dat_intercept[, estimate_yj := car::yjPower(estimate, -0.1)]
dat_intercept[, estimate_boxcox := car::bcPower(estimate, -0.1)]

df_bc <- data.frame(dat_intercept[estimate > 0])
MASS::boxcox(lm(estimate ~ 1, data = df_bc)) %>% as.data.table
car::bcPower(df_bc$estimate, lambda = -0.1) %>% qplot
car::yjPower(dat_intercept$estimate, lambda = -0.1) %>% qplot
library(bestNormalize)
yj_intercept <- yeojohnson(dat_intercept$estimate)
dat_intercept[, estimate_yj := predict(yj_intercept)]

gm1_elev <- gam(estimate_yj ~ s(elev, k = 10), data = dat_intercept)
gm1_elev
gam.check(gm1_elev)
plot(gm1_elev)
plot(fitted(gm1_elev), resid(gm1_elev))

dat_intercept[, gm1_fit_yj := fitted(gm1_elev)]
dat_intercept[, gm1_fit := predict(yj_intercept, newdata = gm1_fit_yj, inverse = T)]
dat_intercept %>% 
  ggplot(aes(elev))+
  geom_line(aes(y = gm1_fit))+
  geom_point(aes(y = estimate))

# -> looks good

# test gam ----------------------------------------------------------------


gm1_elev <- gam(estimate_yj ~ s(elev, k = 10), data = dat_intercept)
gm1_elev
gam.check(gm1_elev)
plot(gm1_elev)
plot(fitted(gm1_elev), resid(gm1_elev))

# dat_intercept[, gm1_fit_yj := fitted(gm1_elev)]
# dat_intercept[, gm1_fit := predict(yj_intercept, newdata = gm1_fit_yj, inverse = T)]
# dat_intercept %>% 
#   ggplot(aes(elev))+
#   geom_line(aes(y = gm1_fit))+
#   geom_point(aes(y = estimate))

# -> looks good



gm2_lonlat <- gam(estimate_yj ~ s(lon, lat), data = dat_intercept)
vis.gam(gm2_lonlat, plot.type = "contour")
gam.check(gm2_lonlat)



# test s, te, ti
kk <- 8

gm3_lle_1a <- gam(estimate_yj ~ s(lon, k = kk) + s(lat, k = kk) + ti(lon, lat, k = c(8,8)) + s(elev), data = dat_intercept)
vis.gam(gm3_lle_1a, plot.type = "contour")

gm3_lle_1b <- gam(estimate_yj ~ s(lon, k = kk) + s(lat, k = kk) + s(elev, k = kk) +
                    ti(lon, lat, elev, k = c(kk,kk,kk)), data = dat_intercept)
gm3_lle_1b
vis.gam(gm3_lle_1b, plot.type = "contour")


gm3_lle_1 <- gam(estimate_yj ~ s(lon, lat, k = 8*8, bs = "tp") + s(elev), data = dat_intercept)
gm3_lle_1
gam.check(gm3_lle_1)
plot(gm3_lle_1)
vis.gam(gm3_lle_1, plot.type = "contour")

gm3_lle_2 <- gam(estimate_yj ~ te(lon, lat, k = c(8,8), bs = "tp") + s(elev), data = dat_intercept)
gm3_lle_2
gam.check(gm3_lle_2)
plot(gm3_lle_2)
vis.gam(gm3_lle_2, plot.type = "contour")

gm3_lle_1 <- gam(estimate_yj ~ s(lon, lat, elev, bs = "tp") + s(elev), data = dat_intercept)
gm3_lle_1
gam.check(gm3_lle_1)
plot(gm3_lle_1)
vis.gam(gm3_lle_1, plot.type = "contour")



cbind(AIC(gm1_elev, gm2_lonlat, gm3_lle_1, gm3_lle_2),
      ll = sapply(list(gm1_elev, gm2_lonlat, gm3_lle_1, gm3_lle_2), logLik))


# test s, but scale lle

dat_intercept[, 
              paste0(c("lon", "lat", "elev"), "_scaled") := lapply(.SD, scale),
              .SDcols = c("lon", "lat", "elev")]
gm3_raw <- gam(estimate_yj ~ s(lon, elev), data = dat_intercept)
gm3_raw
gm3_scaled <- gam(estimate_yj ~ s(lon_scaled, elev_scaled), data = dat_intercept)
gm3_scaled
plot(gm3_raw)
plot(gm3_scaled)
vis.gam(gm3_raw, plot.type = "contour")
vis.gam(gm3_scaled, plot.type = "contour")

# -> need to scale if combine latlon with elev


# smi piemonte vda --------------------------------------------------------


sf_meta_hs <- sf::st_as_sf(join_meta_hs[startsWith(provider, "IT")], 
                           coords = c("Longitude", "Latitude"), crs = sf::st_crs(4326))
sf_meta_hs$provider2 <- sf_meta_hs$provider == "IT_SMI"
mapview::mapview(sf_meta_hs, zcol = "provider2")


meta_it_vda$Name %>% sort
meta_it_piemonte$Name %>% sort

smi_duplicated <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/01_MERGING/SMI/duplicates_smi_piemonte.txt")
meta_it_piemonte[Name %in% smi_duplicated$Piedmont]
smi_duplicated[!Piedmont %in% meta_it_piemonte$Name]
meta_it_piemonte$Name %>% sort()

meta_it_smi_hn[Name %in% smi_duplicated$SMI]
meta_it_smi_hs[Name %in% smi_duplicated$SMI]

smi_duplicated[! SMI %in% meta_it_smi_hn$Name]

meta_it_smi_hn[Name == "Lago_Codelago_Devero"]
meta_it_smi_hn[startsWith(Name, "Lago_Codelago")]
meta_it_smi_hn[startsWith(Name, "Lago_")]

join_data_hn[provider == "IT_PIEMONTE" & variable %in% smi_duplicated$Piedmont, .N, variable]
join_data_hs[provider == "IT_PIEMONTE" & variable %in% smi_duplicated$Piedmont, .N, variable]



# stns meta == data -------------------------------------------------------

stns_data_hn <- unique(join_data_hn$variable)
missing_meta_hn <- sort(stns_data_hn[! stns_data_hn %in% join_meta_hn$Name])
missing_meta_hn_new <- sort(join_meta_hn[provider == "CH_METEOSWISS" & grepl("na", Name), Name])

names(missing_meta_hn_new) <- missing_meta_hn

join_data_hn[variable %in% missing_meta_hn, .N, .(variable, provider)]

# join_meta_hn[Name %in% stns_data_hn]

cbind(sort(missing_meta_hn),
      sort(join_meta_hn[provider == "CH_METEOSWISS" & grepl("na", Name), Name]))


# some plot backup --------------------------------------------------------


dt_year0_nonan[mw_start_year == 1980 & qprob == 0.5 & snow == "HS"] %>% 
  ggplot(aes(Elevation, estimate, shape = country))+
  geom_hline(yintercept = 0)+
  # geom_point(size = 0.5)+
  # geom_smooth(se = F)+
  geom_bin2d(bins = 50)+
  scale_fill_viridis_c(trans = "log10")+
  coord_cartesian(ylim = c(-5, 5))+
  # facet_grid(country ~ month_f)+
  facet_wrap(~month_f, nrow = 2)+
  theme_bw()+
  xlab("Elevation [m]")+
  ylab("Trend mean monthly HS [cm/year]")+
  ggtitle("Linear trend (1980-2000) mean monthly HS")



# check cv gapfill --------------------------------------------------------

dd <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/rds/cv-01-1day.rds")
dd
dd[, .N, Name]
dd[is.na(i_rep)]
dd[!is.na(i_rep),
   .(mae = mean(abs(value_true - value_fill))), 
   .(month, Name)] %>% 
  dcast(Name ~ month)


# cv gapfill 5 days ----------------------------------------------------------

run_na <- sum_run(!is.na(mat_1[, i_stn]), k = 5, na_pad = T)

i_row_possible <- which(run_na == 5 & month(vec_dates) %in% c(11, 12, 1:5))

if(length(i_row_possible) == 0) return(NULL)

if(length(i_row_possible) < 100){
  i_row_sample <- i_row_possible
} else {
  i_row_sample <- sample(i_row_possible, 100)
}


vec_dates[i_row_sample[1] - 4:0]

mat_1[i_row_sample[1] - 4:0, i_stn]



# check save refg gapfill -------------------------------------------------

library(fs)
dir_ls("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/04_GAPFILL/aux-ref-parameter/cv-1day/",
       recurse = T,
       regexp = "csv")


# try reading in grib -----------------------------------------------------

library(raster)
library(rgdal)
gdalDrivers()

rgd <- readGDAL("~/projects-jupyter/cdsapi/era5-test-cdo.grib")
rgd
plot(rgd)
# works

# sf1 <- sf::read_sf("~/projects-jupyter/cdsapi/era5-test-cdo.grib")
# no

library(stars)
st1 <- read_stars("~/projects-jupyter/cdsapi/era5-test-cdo.grib")
st1
plot(st1)
st_crs(st1)
# works

rr <- raster("~/PycharmProjects/cds/download/era5-test.grib")
rr <- raster("~/projects-jupyter/cdsapi/era5-test-cdo.grib")
# no


# convert grib to netcdf first using cdo
# 'cdo -f nc copy <in> >out>
rr <- raster("~/projects-jupyter/cdsapi/era5-test-grib2nc.nc")
plot(rr)

st2 <- read_stars("~/projects-jupyter/cdsapi/era5-test-grib2nc.nc")
st2
plot(st2)

# try extract points
dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/meta_wide_HS.rds")
dat_meta_sub <- dat_meta[sample.int(.N, 20)]
sf_meta_sub <- st_as_sf(dat_meta_sub,
                        coords = c("Longitude", "Latitude"),
                        crs = st_crs("+proj=longlat +datum=WGS84"))
mapview::mapview(sf_meta_sub)

st_crs(st2)
st2[sf_meta_sub]


library(eurocordexr)
dt_nc <- nc_grid_to_dt("~/projects-jupyter/cdsapi/era5-test-grib2nc.nc", 
                                    "var130",
                                    add_xy = T)
dt_nc %>% str
library(ggplot2)
ggplot(dt_nc, aes(lon, lat, fill = var130))+geom_raster()

# dt_point <- rotpole_nc_point_to_dt("~/projects-jupyter/cdsapi/era5-test-grib2nc.nc",
#                                    variable = "var130",
#                                    point_lat = 46.96,
#                                    point_lon = 11.34)
# nc_open("~/projects-jupyter/cdsapi/era5-test-grib2nc.nc")
# -> does not work, just use the dt and manual dist lon lat

# chelsa wworks!
rr <- raster("/mnt/CEPH_PROJECTS/CLIRSNOW/chelsa/prec/CHELSA_prec_1979_01_V1.2.1.tif")
rr
rr[rr > 60000] <- NA
plot(rr)

rr2 <- crop(rr, extent(0, 20, 40, 50))
rr2
plot(rr2)

# st_chel <- read_stars("/mnt/CEPH_PROJECTS/CLIRSNOW/chelsa/prec/CHELSA_prec_1979_01_V1.2.1.tif")
# st_chel
# 
# sub_with_st <- st_chel[sf_meta_sub]
# plot(sub_with_st)
# sub_with_st %>% str



nc_open("/mnt/CEPH_PROJECTS/CLIRSNOW/laprec/LAPrec1901.v1.0.nc")
rr <- raster("/mnt/CEPH_PROJECTS/CLIRSNOW/laprec/LAPrec1901.v1.0.nc")
rr
plot(rr)


nc_open("/mnt/CEPH_PROJECTS/CLIRSNOW/zz_eobs/v20.0e/rr_ens_mean_0.1deg_reg_v20.0e.nc")

# summary
# chelsa: raster
# grib: cdo (to nc + monmean/sum + sellatlon) : eurocordexr
# laprec: raster (since laea)
# eobs: cdo monmean + sellatlon : eurocordexr
# uerra/mescan: ??




# overview seasonal -------------------------------------------------------

rbind(
  dat_all_OctSep[, 
                 .(hs = sum(!is.na(HS)),
                   hn = sum(!is.na(HN)),
                   seas = "oct-sep"), 
                 hydro_year],
  dat_all_NovMay[, 
                 .(hs = sum(!is.na(HS)),
                   hn = sum(!is.na(HN)),
                   seas = "nov-may"), 
                 hydro_year]
) %>% 
  ggplot(aes(hydro_year, hn, colour = seas))+
  geom_point()+
  geom_line()


# aux grid data test ------------------------------------------------------

# summary
# chelsa: raster
# grib: cdo (to nc + monmean/sum + sellatlon) : eurocordexr
# laprec: raster (since laea)
# eobs: cdo monmean + sellatlon : eurocordexr
# uerra/mescan: ??




rr0 <- raster("/mnt/CEPH_PROJECTS/CLIRSNOW/uerra/total_precipitation/mescan_tp_1961.grib")
rr0

st_bbox(bbox(matrix(c(3,18,42,50), nrow = 2)))
extent(3,18,42,50) %>% 
  st_bbox(crs = st_crs(4326)) %>% 
  st_as_sfc() %>% 
  st_transform(st_crs(rr0))


# check hzb seasonal data with daily --------------------------------------


hzb_meta <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/06_SEASONAL/from-non-daily-data/hzb/meta.csv")
hzb_data <- fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/06_SEASONAL/from-non-daily-data/hzb/data.csv")

hzb_data %>% summary
hzb_past_stns <- sort(unique(hzb_data$stn_name_safe))
dat_all_OctSep[Name %in% hzb_past_stns, .N, Name]
meta_all[Name %in% hzb_past_stns]
meta_all[Provider == "AT_HZB"]

# -> many stations in past HZB, which are not in daily data (and vice versa)

hzb_data[stn_name_safe %in% meta_all$Name] %>% summary
hzb_data[! stn_name_safe %in% meta_all$Name] %>% summary
# -> contains also data after 1970, try to merge and see if differences?

dat_check_hzb <- merge(
  dat_all_OctSep[Name %in% meta_all[Provider == "AT_HZB", Name],
                 .(Name, hydro_year, sum_hn_daily = HN, in_daily = "in_daily")],
  hzb_data[, .(Name = stn_name_safe, hydro_year, sum_hn_seasonal = sum_hn, in_seasonal = "in_seasonal")],
  by = c("Name", "hydro_year"), all = T
)

with(dat_check_hzb, table(in_daily, in_seasonal, useNA = "a"))
dat_zz <- dat_check_hzb[, .N, .(Name, in_daily, in_seasonal)]
with(dat_check_hzb, plot(sum_hn_daily, sum_hn_seasonal))

# -> can use both, just merge the data


# compare stns subs for eof and gapfill -----------------------------------

dat_hs_gp <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/data_long_HS.rds")
dat_meta_gp <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/05_MONTHLY/rds/1961-2020_gapfilled_HS_sub/meta_long_HS.rds")

dat_hs_sub2$Name %>% unique %>% sort -> stn_sub_eof


dat_meta_gp[Name %in% stn_sub_eof]




# seasonal lm --------------------------------------------------------------


# dat_hs2 <- dat_hs[year <= 2019 & month %in% c(12, 1, 2) & !is.na(HS)]
dat_hs2 <- dat_hs[year >= 1971 & year <= 2019 & month %in% c(12, 1, 2) & !is.na(HS)]
with(dat_hs2, table(year, month))

dat_hs2[, nn_year := .N, .(Name, month)]
dat_hs_full <- dat_hs2[nn_year == max(nn_year)]

dat_meta[Name %in% dat_hs_full$Name, .N, Provider]


# lm


mitmatmisc::add_season_fct(dat_hs)

dat_hs_season <- dat_hs[, .(HS = mean(HS), nn = .N), .(Name, season, hydro_year)]
dat_hs_season <- dat_hs_season[nn == 3]

dat_hs_season
with(dat_hs_season[!is.na(HS)], table(hydro_year, season))


dat_hs_season2 <- dat_hs_season[hydro_year >= 1981 & hydro_year <= 2017 & !is.na(HS)]
with(dat_hs_season2, table(hydro_year, season))

dat_hs_season2[, nn_year := .N, .(Name, season)]
dat_hs_season_full <- dat_hs_season2[nn_year == max(nn_year)]

dat_hs_season_full[, length(unique(Name)), season]

dat_meta[Name %in% dat_hs_season_full$Name, .N, Provider]


# some plots clim ---------------------------------------------------------




dat_clim %>% 
  melt(measure.vars = c("HS", "tmean_lapse", "prec")) %>% 
  merge(dat_meta_clust, by = "Name") -> dat_plot

# 
# dat_plot %>% 
#   ggplot(aes(value, Elevation, color = cluster_fct))+
#   geom_point()+
#   # geom_smooth(se = F)+
#   scale_color_brewer(palette = "Set1")+
#   theme_bw()+
#   facet_grid(season ~ variable, scales = "free_x")


dat_plot %>%
  ggplot(aes(Elevation, value, color = cluster_fct))+
  geom_point()+
  # geom_smooth(se = F)+
  scale_color_brewer(palette = "Set1")+
  theme_bw()+
  facet_grid(variable ~ season, scales = "free_y")


dat_xlim <- dat_plot[, .(max_elev = max(Elevation)), .(cluster_fct)] 
setorder(dat_xlim, cluster_fct)
dat_xlim[, max_elev := c(1250, 1250, 3000, 3000, 1250)]
# 
# dat_plot[season == "DJF"] %>% 
#   ggplot(aes(Elevation, value, color = cluster_fct))+
#   geom_point()+
#   # geom_smooth(se = F, method = lm)+
#   scale_color_brewer(palette = "Set1")+
#   theme_bw()+
#   theme(strip.placement = "outside",
#         legend.position = "none")+
#   facet_grid(variable ~ cluster_fct, scales = "free", space = "free_x", switch = "y")+
#   scale_x_continuous(breaks = seq(0, 3000, by = 500), limits = c(0, NA))+
#   geom_blank(inherit.aes = F, data = dat_xlim, aes(x = max_elev))+
#   xlab("Elevation [m]")+
#   ylab(NULL)
# 
# ggsave("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/Figure 7.png",
#        width = 12, height = 6)
# 

# dat_plot[, cluster_fct2 := fct_collapse(cluster_fct,
#                                         North = c("low NE", "low NW", "high N"),
#                                         South = c("SW", "SE"))]
# 
# dat_plot[, cluster_fct3 := fct_collapse(cluster_fct,
#                                         High = c("SW", "high N"),
#                                         Low = c("low NW", "low NE", "SE"))]




# trend by clim -----------------------------------------------------------

# try scatter of 71-19 trends versus climate

dat_lm <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-03-full_1971-2019-calyear.rds")


dat_lm[term == "year0" & !is.na(statistic)] %>% 
  merge(dat_clim[season == "DJF"], by = "Name") %>% 
  merge(dat_meta_clust, by = "Name") -> dat_plot_lm

mitmatmisc::add_month_fct(dat_plot_lm, 10)

dat_plot_lm %>% 
  ggplot(aes(HS, estimate, colour = cluster_fct))+
  geom_point()+
  scale_color_brewer(palette = "Set1")+
  # facet_grid(cluster_fct ~ month_fct)+
  facet_wrap(~month_fct)+
  theme_bw()

zz1 <- dat_plot_lm[month == 3 & cluster_fct == "SW" & HS > 75]
setorder(zz1, estimate)
zz1

dat_plot_lm %>% 
  ggplot(aes(prec, estimate, colour = cluster_fct))+
  geom_point()+
  scale_color_brewer(palette = "Set1")+
  # facet_grid(cluster_fct ~ month_fct)+
  facet_wrap(~month_fct)+
  theme_bw()







# try to fill spring HS for spring SCD (and maxHS) ------------------------



stn_not <- dat_check2[hydro_year==2019, Name]
all_stn <- unique(dat_hs_sub$Name)
all_stn[! all_stn %in% stn_not]

dat_hs_sub[Name == "Zurndorf"] %>% summary

dat_hs_sub[Name == "Zurndorf", sum(is.na(HS)), .(month(date), hydro_year)] %>% 
  dcast(hydro_year ~ month)

dat_zz <- dat_hs_sub[Name == "Zurndorf" & hydro_year == 2019]

# -> not needed


# calc monthly and seasonal means based on EOF results --------------------



library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(foreach)



load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/regions-01-sinkr-eof.rda")
n_pc <- 20

sinkr_eof %>% str

mat_eof %>% str
mat_test <- sinkr_eof$A[, 1:n_pc] %*% t(sinkr_eof$u[, 1:n_pc])
mat_test2 <- scale(mat_test, F, 1/sinkr_eof$F1_scale)
mat_test2 <- scale(mat_test2, -1 * sinkr_eof$F1_center, F)

# summary(as.vector(mat_eof - mat_test2))
# qplot(as.vector(mat_eof - mat_test2))

i_sample <- sample(length(mat_eof), 1e4)
i_sample_na <- !is.na(mat_eof[i_sample])
plot(mat_eof[i_sample][i_sample_na], mat_test2[i_sample][i_sample_na])

# -> ok!

sinkr_eof_summary
summary_sinkr_eof(sinkr_eof, 20)

mat_test2 %>% str

dat_hs_eof <- as.data.table(mat_test2)
setnames(dat_hs_eof, colnames(mat_eof))
# dat_hs_eof[, date := ]


dat_hs_out[abs(HS - HS_eof) > 300]

dat_hs_out[abs(HS - HS_eof) > 100, .N, Name] -> dat_zz



# compare eof monthly to "normal" agg
min_frac_avail <- 0.9
dat_month_clim <- dat_hs_out[, .(HS = mean(HS, na.rm = T),
                                 nn_HS = sum(!is.na(HS)),
                                 HS_eof = mean(HS_eof, na.rm = T),
                                 nn_HS_eof = sum(!is.na(HS_eof)),
                                 nn_in_month = days_in_month(date[1])),
                             .(Name, month(date))]

dat_month_clim[nn_HS < min_frac_avail * nn_in_month * 30, HS := NA]
dat_month_clim[nn_HS_eof < min_frac_avail * nn_in_month * 30, HS_eof := NA]


dat_month_clim %>% summary
dat_month_clim %>% 
  ggplot(aes(HS, HS_eof))+
  geom_abline()+
  geom_point()+
  facet_wrap(~month)

dat_month_clim %>% 
  ggplot(aes(HS, HS_eof - HS))+
  geom_point()+
  facet_wrap(~month)

dat_month_clim[HS > 5] %>% 
  ggplot(aes(HS, (HS_eof - HS)/HS))+
  geom_point()+
  facet_wrap(~month)


# check clim eof raw ------------------------------------------------------



dat_hs_eof <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/data-03-daily-from-eof.rds")
dat_meta_cluster <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/NIMBUS/rds/meta-with-cluster-01.rds")

# calc clim
min_frac_avail <- 0.9
dat_month_clim <- dat_hs_eof[, .(HS = mean(HS, na.rm = T),
                                 nn_HS = sum(!is.na(HS)),
                                 HS_eof = mean(HS_eof, na.rm = T),
                                 nn_HS_eof = sum(!is.na(HS_eof)),
                                 nn_in_month = days_in_month(date[1])),
                             .(Name, month(date))]

dat_month_clim[nn_HS < min_frac_avail * nn_in_month * 30, HS := NA]
dat_month_clim[nn_HS_eof < min_frac_avail * nn_in_month * 30, HS_eof := NA]


dat_plot <- dat_month_clim %>% 
  merge(dat_meta_cluster, by = "Name")

mitmatmisc::add_month_fct(dat_plot, 10)

dat_plot[!is.na(HS)] %>% 
  ggplot(aes(Longitude, Latitude, colour = HS))+
  borders()+
  geom_point()+
  facet_wrap(~month_fct)+
  # scale_color_scico(palette = "batlow", direction = -1)+
  scale_color_viridis_c(direction = -1)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_quickmap(xlim = range(dat_meta_cluster$Longitude), ylim = range(dat_meta_cluster$Latitude))+
  theme_bw()


dat_plot %>% 
  ggplot(aes(Longitude, Latitude, colour = HS_eof))+
  borders()+
  geom_point()+
  facet_wrap(~month_fct)+
  # scale_color_scico(palette = "batlow", direction = -1)+
  scale_color_viridis_c(direction = -1)+
  xlab(NULL)+ylab(NULL)+
  scale_x_continuous(labels = function(x) paste0(x, "° E"))+
  scale_y_continuous(labels = function(x) paste0(x, "° N"))+
  coord_quickmap(xlim = range(dat_meta_cluster$Longitude), ylim = range(dat_meta_cluster$Latitude))+
  theme_bw()



# clim of uerra t2m and prec ----------------------------------------------




library(raster)
library(stars)

rs_modis <- read_stars("/mnt/CEPH_PROJECTS/SNOW_3rd_gen/CLOUDREMOVAL/v1.1/annual_SCD/2000-02-24_2000-09-30.tif")

files_t2m <- paste0("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/08_AUX_GRIDS/monthly/UERRA_t2m/mescan_t2m_",
                    1981:2010, ".nc")
rs_t2m1 <- read_stars(files_t2m[1])
rs_t2m <- read_stars(files_t2m)

rs_modis %>% 
  st_bbox() %>%
  st_as_sfc %>% 
  st_transform(st_crs(rs_t2m)) %>% 
  st_bbox -> bb_new

rs_t2m_alps <- st_crop(rs_t2m, bb_new)

zz %>% 
  st_dimensions("time")
rs_t2m %>% 
  st_get_dimension_values("time") %>% 
  month() -> dim_time_month


rs_t2m_alps %>% 
  dplyr::slice(time, which(dim_time_month %in% c(11,12,1:5))) %>% 
  st_apply(c("x", "y", "height"), mean) -> rs_t2m_alps_clim



ggplot()+
  geom_stars(data = rs_t2m_alps_clim)


# EOF ---------------------------------------------------------------------


