# read in all hs and hn data

library(fs)
library(data.table)
library(magrittr)
library(readxl)
library(lubridate)



# meta --------------------------------------------------------------------


dat_meta <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/meta-01-original.rds")

# raw slovenia needed, since columns in data are id not name
dat_meta_slovenia <- rbind(
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/SLOVENIA/List_stations_HN.csv", encoding = "Latin-1"),
  fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/SLOVENIA/List_stations_HS.csv", encoding = "Latin-1")
) %>% unique

source("R/functions/safe_names.R")
dat_meta_slovenia[, stn_name_safe := stringi::stri_trans_totitle(safe_names(tolower(NAME)))]
dat_meta_slovenia[stn_name_safe %in% c("Bilje", "Novo_mesto", "Celje_medlog", "Ratece","Murska_sobota_rakican"),
                  stn_name_safe := paste0(stn_name_safe, "_", ID)]

# data hs -----------------------------------------------------------------

source("R/functions/safe_names.R")

f_read_plus <- function(fn){
  dat <- fread(fn)
  dat2 <- melt(dat, id.vars = "Date", variable.factor = F)
  dat2[, Date := ymd(Date)]
  dat2
}

l_hs <- list(
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/AT_HZB/read_HS.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/BOLZANO/read_manualHS.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/CH_METEOSWISS/read_manualHS.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/CH_SLF/read_manualHS.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/FR_METEOFRANCE/read_manualHS.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/GERMANY/read_HS.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/TRENTINO/read_manualHS.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/PIEMONTE/read_manualHS.csv")
)
names(l_hs) <- c("AT_HZB", "IT_BZ", "CH_METEOSWISS", "CH_SLF", "FR_METEOFRANCE", 
                 "DE_DWD", "IT_TN", "IT_PIEMONTE")


fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/VDA/read_manualHS_1910_1960.csv", 
      encoding = "Latin-1") %>% 
  melt(id.vars = "Date", variable.factor = F) -> dat_vda1
fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/VDA/read_manualHS_1960_2010.csv", 
      encoding = "Latin-1") %>% 
  melt(id.vars = "Date", variable.factor = F) -> dat_vda2
dat_vda <- rbind(dat_vda1, dat_vda2)
dat_vda[, Date := ymd(Date)]
dat_vda[, variable := safe_names(variable)]
summary(dat_vda)

fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/VENETO/read_manualHS.csv") %>% 
  melt(id.vars = "Date", variable.factor = F) -> dat_veneto
dat_veneto[, Date := dmy(Date)]
summary(dat_veneto)

fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/FVG/read_AINEVA_HS.csv") %>% 
  melt(id.vars = "Date", variable.factor = F) -> dat_fvg
dat_fvg[, Date := dmy(Date)]
dat_fvg[value >= 800 | value < 0, value := NA]
summary(dat_fvg)

fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/LOMBARDIA/read_manual_HS.csv") %>% 
  melt(id.vars = "date", variable.factor = F) -> dat_lombardia
setnames(dat_lombardia, "date", "Date")
dat_lombardia[, Date := dmy(Date)]
summary(dat_lombardia)

fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/SLOVENIA/read_manualHS.csv", header = T) %>% 
  melt(id.vars = "Date", variable.factor = F) -> dat_slovenia
dat_slovenia[, Date := ymd(Date)]
dat_slovenia[value == -1, value := 0] # days without snow? 
summary(dat_slovenia)
dat_slovenia[, variable := as.numeric(variable)]
dat_slovenia <- merge(dat_slovenia, dat_meta_slovenia[, .(variable = ID, stn_name_safe)])
dat_slovenia[, variable := NULL]
setnames(dat_slovenia, "stn_name_safe", "variable")

l_hs2 <- list(dat_fvg, dat_lombardia, dat_slovenia, dat_vda, dat_veneto)
names(l_hs2) <- c("IT_FVG", "IT_LOMBARDIA", "SI_ARSO", "IT_VDA", "IT_VENETO")

dat_hs <- rbindlist(c(l_hs, l_hs2), use.names = T, fill = T, idcol = "provider")
setnames(dat_hs, c("provider", "date", "stn_name", "hs"))
dat_hs2 <- dat_hs[!is.na(hs)]

saveRDS(dat_hs2, file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hs-01-original.rds")
rm(l_hs, l_hs2, dat_hs)



# data hn -----------------------------------------------------------------

source("R/functions/safe_names.R")

f_read_plus <- function(fn){
  dat <- fread(fn)
  dat2 <- melt(dat, id.vars = "Date", variable.factor = F)
  dat2[, Date := ymd(Date)]
  dat2
}

l_hn <- list(
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/AT_HZB/read_HN.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/BOLZANO/read_manualHN.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/CH_METEOSWISS/read_manualHN.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/CH_SLF/read_manualHN.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/FR_METEOFRANCE/read_manualHN.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/GERMANY/read_HN.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/TRENTINO/read_manualHN.csv"),
  f_read_plus("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/PIEMONTE/read_manualHN.csv")
)
names(l_hn) <- c("AT_HZB", "IT_BZ", "CH_METEOSWISS", "CH_SLF", "FR_METEOFRANCE", 
                 "DE_DWD", "IT_TN", "IT_PIEMONTE")

# vda no HN?
# fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/VDA/", encoding = "Latin-1") %>% 
#   melt(id.vars = "Date", variable.factor = F) -> dat_vda
# dat_vda[, Date := dmy(Date)]
# dat_vda[, variable := safe_names(variable)]
# summary(dat_vda)

fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/VENETO/read_manualHN.csv",
      na.strings = c("NA", "///")) %>% 
  melt(id.vars = "Date", variable.factor = F) -> dat_veneto
dat_veneto[, Date := dmy(Date)]
summary(dat_veneto)

fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/FVG/read_AINEVA_HN.csv") %>% 
  melt(id.vars = "Date", variable.factor = F) -> dat_fvg
dat_fvg[, Date := dmy(Date)]
dat_fvg[startsWith(as.character(value), "9"), table(value)]
dat_fvg[startsWith(as.character(value), "8"), table(value)]
dat_fvg[value == 999 | value == 899, value := 0] # traces of HN or traces of HN with rain
dat_fvg[value == 800 , value := 0] # rain on snow, no fresh snow
dat_fvg[value > 800, value := value - 800]
dat_fvg[value < 0, value := NA]
summary(dat_fvg)

fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/LOMBARDIA/read_manual_HN.csv") %>% 
  melt(id.vars = "date", variable.factor = F) -> dat_lombardia
setnames(dat_lombardia, "date", "Date")
dat_lombardia[, Date := dmy(Date)]
dat_lombardia[value > 800, value := NA]
summary(dat_lombardia)

fread("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/ORIGINAL_DATA/SLOVENIA/read_manualHN.csv", header = T) %>% 
  melt(id.vars = "Date", variable.factor = F) -> dat_slovenia
dat_slovenia[, Date := ymd(Date)]
dat_slovenia[value == -1, value := 0] # days without snow?
summary(dat_slovenia)
dat_slovenia[, variable := as.numeric(variable)]
dat_slovenia <- merge(dat_slovenia, dat_meta_slovenia[, .(variable = ID, stn_name_safe)])
dat_slovenia[, variable := NULL]
setnames(dat_slovenia, "stn_name_safe", "variable")

# l_hn2 <- list(dat_fvg, dat_lombardia, dat_slovenia, dat_vda, dat_veneto)
# names(l_hn2) <- c("IT_FVG", "IT_LOMBARDIA", "SI_ARSO", "IT_VDA", "IT_VENETO")
l_hn2 <- list(dat_fvg, dat_lombardia, dat_slovenia, dat_veneto)
names(l_hn2) <- c("IT_FVG", "IT_LOMBARDIA", "SI_ARSO", "IT_VENETO")

dat_hn <- rbindlist(c(l_hn, l_hn2), use.names = T, fill = T, idcol = "provider")
setnames(dat_hn, c("provider", "date", "stn_name", "hn"))
dat_hn2 <- dat_hn[!is.na(hn)]

saveRDS(dat_hn2, file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/OVERVIEW/r-data/hn-01-original.rds")
rm(l_hn, l_hn2, dat_hn)




