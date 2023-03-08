# QC finalize all steps
# and save data


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(forcats)
library(directlabels)
library(readxl)
library(writexl)
library(fs)

# prep data ---------------------------------------------------------------



dat_meta_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/meta_long_HN_HS.rds")
dat_1 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1961-2020/r-data/data_long_HN_HS.rds")

dat_meta_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/meta_long_HN_HS.rds")
dat_2 <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/02_JOIN_REGIONS/1787hn_1879hs-1960/r-data/data_long_HN_HS.rds")


dat_meta_all <- unique(rbind(dat_meta_1, dat_meta_2))
setkey(dat_meta_all, Provider, Name)

dat_all <- rbind(dat_2, dat_1)
setkey(dat_all, Name, Date)



dat_all[, HN_lag := shift(HN, type = "lag"), .(Name)]
dat_all[, HS_lag := shift(HS, type = "lag"), .(Name)]
dat_all[, HS_diff := HS - HS_lag]



# make integer ---------------------------------------------------------------

is.wholenumber <- function(x, tol = .Machine$double.eps^0.5)  abs(x - round(x)) < tol

dat_all[!is.wholenumber(HS), .N, .(Provider, Name)]
dat_all[Name == "Arabba" & !is.wholenumber(HS)]
dat_all[Name == "Gurtnellen_CH_SLF" & !is.wholenumber(HS)]
dat_all[Name == "Gressoney_La_Trinite_Eselbode" & !is.wholenumber(HS)]

# -> ok for HS

dat_all[!is.wholenumber(HN), .N, .(Provider, Name)]
dat_all[Name == "Falcade" & !is.wholenumber(HN)]
dat_all[Name == "Torino" & !is.wholenumber(HN)]
dat_all[Provider == "CH_SLF" & !is.wholenumber(HN), table(HN)]

# -> ok also for HN
dat_all[, HN := as.integer(round(HN))]
dat_all[, HS := as.integer(round(HS))]


# non-zero ----------------------------------------------------------------

dat_all[HN < 0]
dat_all[HS < 0]

dat_all[HS < 0, HS := NA]

# maximum daily HN --------------------------------------------------------

# dat_all[HN > 200]
# 
# dat_all[Name == "Meteomont_Loc_Planpincieux"] %>% 
#   .[Date >= "2007-01-10" & Date <= "2007-02-01"]
# 
# dat_all[Name == "Schwagalp_CH_METEOSWISS"] %>% 
#   .[Date >= "2012-02-01" & Date <= "2012-03-01"]
# 
# dat_all[HN > 200, HN := NA]

# temporal consistency ----------------------------------------------------

dat_tc <- read_excel("manual-qc/v02/temporal-consistency_prefilled-v01_filled.xlsx")
setDT(dat_tc)
dat_tc %>% str
dat_tc[, Date_error := as.Date(Date_error)]
dat_tc[, Date := as.Date(Date)]
setkey(dat_tc, Name, Date)



dat_tc[HN_error == 1]
dat_all[dat_tc[HN_error == 1], HN := NA]
# dat_all[.("Alvaneu_Dorf", ymd("2016-03-05"))]
# dat_all[.("Ceresole_Reale", ymd("2006-03-17"))]

dat_tc[HS_error == 1]
dat_all[dat_tc[HS_error == 1], HS := NA]
# dat_all[.("Abries_rm", ymd("2019-01-09"))]
# dat_all[.("Airolo", ymd("1973-06-30"))]



# zero-NA -----------------------------------------------------------------

dat_0na_overview <- read_excel("manual-qc/v02/zero-NA-overview_prefilled-v01_filled.xlsx")
setDT(dat_0na_overview)
files_sub <- fs::path("manual-qc/v02/zero-NA_prefilled-v01_filled/", 
                      dat_0na_overview[zero_NA == 1, Name], ext = "xlsx")
l <- lapply(files_sub, read_excel)
names(l) <- dat_0na_overview[zero_NA == 1, Name]
dat_0na <- rbindlist(l, fill = T, use.names = T, idcol = "Name")
dat_0na[, Date := as.Date(Date)]
setkey(dat_0na, Name, Date)

dat_0na[zeroNA == 1 & HS == 0]
dat_all[dat_0na[zeroNA == 1 & HS == 0], HS := NA]
# dat_all[.("Barcelonnette", ymd("1971-10-01"))]
# dat_all[.("Watzmannhaus", ymd("1953-09-29"))]

dat_0na[remove_too == 1]
dat_all[dat_0na[remove_too == 1], HS := NA]
# dat_all[.("Bellinzona_CH_METEOSWISS", ymd("2008-12-27"))]




# suspicious and to check series ------------------------------------------

dat_sac_overview <- read_excel("manual-qc/v02/series-to-check-overview_prefilled-v01_filled.xlsx")
setDT(dat_sac_overview)
files_sub <- fs::path("manual-qc/v02/series-to-check_prefilled-v01_filled/", 
                      dat_sac_overview[remove_some_values == 1, Name], ext = "xlsx")
l <- lapply(files_sub, read_excel)
names(l) <- dat_sac_overview[remove_some_values == 1, Name]
dat_sac <- rbindlist(l, fill = T, use.names = T, idcol = "Name")
dat_sac[, Date := as.Date(Date)]
setkey(dat_sac, Name, Date)

dat_sac[to_remove == 1]
dat_all[dat_sac[to_remove == 1], HS := NA]
# dat_all[.("Bionaz_Place_Moulin", ymd("2009-05-01"))]
# dat_all[.("Redagno_Osservatore", ymd("1998-06-25"))]


# remove whole series and suspicious ones
stns_to_remove <- unique(dat_sac_overview[remove_whole_series == 1 | suspicious == 1, Name])
dat_all <- dat_all[!.(stns_to_remove)]


# HN consecutive -> remove ------------------------------------------------

dat_nonzero <- dat_all[HS != 0 & HN != 0]


dat_summary <- dat_nonzero[!is.na(HS_diff) & !is.na(HN_lag), 
                           .(nn = .N, 
                             nn_HNc = sum(HS_diff == HN | HS_diff == HN_lag)),
                           .(Provider, Name)]
dat_nonzero[, 
            .(nn_total_hs = sum(!is.na(HS)),
              nn_total_hn = sum(!is.na(HN))),
            .(Provider, Name)] %>% 
  merge(dat_summary) -> dat_summary2

with(dat_summary2, plot(nn, nn_HNc))
dat_summary2[nn == nn_HNc]
# dat_summary[(nn_HNc / nn) > 0.8]

stns_HNc <- dat_summary2[nn == nn_HNc, Name]
dat_all[Name %in% stns_HNc[3],] %>% summary
dat_nonzero[Name %in% stns_HNc[3],]


dat_all[.(stns_HNc), HN := NA]


# HN HS consistency -------------------------------------------------------

# with or without tolerance of 5cm?
dat_check <- dat_all[(HS - HS_lag) > (HN + HN_lag + 5)]
# dat_check <- dat_all[(HS - HS_lag) > (HN + HN_lag)]


dat_check[, .N, .(Provider, Name)] %>% .[, N] %>% qplot()
dat_check[(HN + HN_lag) != 0, .N, .(Provider, Name)] %>% .[, N] %>% qplot()

dat_check[, .N, .(Provider, Name)] %>% .[, N] %>% table
dat_check[, .N, .(Provider)]
dat_check[(HN + HN_lag) == 0, .N, .(Provider, Name)]
dat_check[HN + HN_lag == 0, .N, .(Provider)]

dat_check
dat_check[Provider == "CH_SLF"]

# -> if measurement errors in HN and HS are considered,
#    then maybe not really worth it...



# HN == HS ----------------------------------------------------------------


dat_summary <- dat_nonzero[!is.na(HS_diff) & !is.na(HN_lag), 
                           .(nn = .N, 
                             nn_HS_eq_HN = sum(HS == HN)),
                           .(Name)]
with(dat_summary, plot(nn, nn_HS_eq_HN))
dat_summary[nn == nn_HS_eq_HN]

# -> hmmm




# split and write ---------------------------------------------------------



# past <= 1960
data_wide_hn <- dcast(dat_all[year(Date) <= 1960], 
                      Date ~ Name, 
                      value.var = "HN")
lgl_cols_to_keep <- sapply(data_wide_hn, function(x) !all(is.na(x)))
chr_cols_to_keep <- names(lgl_cols_to_keep)[lgl_cols_to_keep]
out_data_wide_hn <- data_wide_hn[, chr_cols_to_keep, with = F]
out_meta_wide_hn <- dat_meta_all[Name %in% chr_cols_to_keep]

data_wide_hs <- dcast(dat_all[year(Date) <= 1960], 
                      Date ~ Name, 
                      value.var = "HS")
lgl_cols_to_keep <- sapply(data_wide_hs, function(x) !all(is.na(x)))
chr_cols_to_keep <- names(lgl_cols_to_keep)[lgl_cols_to_keep]
out_data_wide_hs <- data_wide_hs[, chr_cols_to_keep, with = F]
out_meta_wide_hs <- dat_meta_all[Name %in% chr_cols_to_keep]



out_data_long <- dat_all[year(Date) <= 1960, .(Name, Date, HN, HS)]
check_all_na <- out_data_long[, .(all_na = all(is.na(c(HN,HS)))), .(Name)]
stopifnot(all(!check_all_na$all_na))
setkey(out_data_long, Name, Date)

out_meta_long <- dat_meta_all[Name %in% unique(out_data_long$Name)]
setkey(out_meta_long, Provider, Name)

path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1787hn_1879hs-1960/"
if(!dir_exists(path_out)) dir_create(path_out)
saveRDS(out_data_long, file = path(path_out, "data_long_HN_HS.rds"))
saveRDS(out_data_wide_hn, file = path(path_out, "data_wide_HN.rds"))
saveRDS(out_data_wide_hs, file = path(path_out, "data_wide_HS.rds"))
saveRDS(out_meta_long, file = path(path_out, "meta_long_HN_HS.rds"))
saveRDS(out_meta_wide_hn, file = path(path_out, "meta_wide_HN.rds"))
saveRDS(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.rds"))
# 
# path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/csv/1787hn_1879hs-1960/"
# # fwrite(out_data_long, file = path(path_out, "data_long_HN_HS.csv")) # -> too big, and not needed
# fwrite(out_data_wide_hn, file = path(path_out, "data_wide_HN.csv"))
# fwrite(out_data_wide_hs, file = path(path_out, "data_wide_HS.csv"))
# # fwrite(out_meta_long, file = path(path_out, "meta_long_HN_HS.csv")) # consistent with no data
# fwrite(out_meta_wide_hn, file = path(path_out, "meta_wide_HN.csv"))
# fwrite(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.csv"))


# present >= 1961
data_wide_hn <- dcast(dat_all[year(Date) >= 1961], 
                      Date ~ Name, 
                      value.var = "HN")
lgl_cols_to_keep <- sapply(data_wide_hn, function(x) !all(is.na(x)))
chr_cols_to_keep <- names(lgl_cols_to_keep)[lgl_cols_to_keep]
out_data_wide_hn <- data_wide_hn[, chr_cols_to_keep, with = F]
out_meta_wide_hn <- dat_meta_all[Name %in% chr_cols_to_keep]

data_wide_hs <- dcast(dat_all[year(Date) >= 1961], 
                      Date ~ Name, 
                      value.var = "HS")
lgl_cols_to_keep <- sapply(data_wide_hs, function(x) !all(is.na(x)))
chr_cols_to_keep <- names(lgl_cols_to_keep)[lgl_cols_to_keep]
out_data_wide_hs <- data_wide_hs[, chr_cols_to_keep, with = F]
out_meta_wide_hs <- dat_meta_all[Name %in% chr_cols_to_keep]



out_data_long <- dat_all[year(Date) >= 1961, .(Name, Date, HN, HS)]
check_all_na <- out_data_long[, .(all_na = all(is.na(c(HN,HS)))), .(Name)]
stopifnot(all(!check_all_na$all_na))
setkey(out_data_long, Name, Date)

out_meta_long <- dat_meta_all[Name %in% unique(out_data_long$Name)]
setkey(out_meta_long, Provider, Name)



path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/rds/1961-2020/"
if(!dir_exists(path_out)) dir_create(path_out)
saveRDS(out_data_long, file = path(path_out, "data_long_HN_HS.rds"))
saveRDS(out_data_wide_hn, file = path(path_out, "data_wide_HN.rds"))
saveRDS(out_data_wide_hs, file = path(path_out, "data_wide_HS.rds"))
saveRDS(out_meta_long, file = path(path_out, "meta_long_HN_HS.rds"))
saveRDS(out_meta_wide_hn, file = path(path_out, "meta_wide_HN.rds"))
saveRDS(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.rds"))
# 
# path_out <- "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/03_QC1/csv/1961-2020/"
# # fwrite(out_data_long, file = path(path_out, "data_long_HN_HS.csv")) # -> too big 1.4GB
# fwrite(out_data_wide_hn, file = path(path_out, "data_wide_HN.csv"))
# fwrite(out_data_wide_hs, file = path(path_out, "data_wide_HS.csv"))
# # fwrite(out_meta_long, file = path(path_out, "meta_long_HN_HS.csv")) # consistent with no data
# fwrite(out_meta_wide_hn, file = path(path_out, "meta_wide_HN.csv"))
# fwrite(out_meta_wide_hs, file = path(path_out, "meta_wide_HS.csv"))





# EOF ---------------------------------------------------------------------


