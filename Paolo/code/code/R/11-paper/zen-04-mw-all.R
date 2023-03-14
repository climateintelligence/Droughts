# plot trends (30y moving window)


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

library(forcats)
library(foreach)
library(flextable)
library(officer)

dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/meta-with-cluster-01.rds")

# choose hydro_year or calendar year
# dat_lm <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-01-mw-hydro-year.rds")
dat_lm  <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/trends-02-mw-calendar-year.rds")

# dat_lm_sub <- dat_lm[term == "year0" & mw_year_start %in% c(1961, 1976, 1990)]
# dat_lm_sub <- dat_lm[term == "year0" & mw_year_start %in% c(1961, 1971, 1981, 1990)]



# check to remove low trend & intercept -----------------------------------

# dat_lm[!is.na(statistic) & mw_year_start == 1961 & term == "year0", estimate] %>% 
#   qplot(binwidth = 0.01)+xlim(-0.5, 0.5)
# 
# dat_lm[!is.na(statistic) & mw_year_start == 1961 & term == "(Intercept)", estimate] %>% 
#   qplot(binwidth = 0.1)+xlim(NA, 10)
# 
# dat_lm[!is.na(statistic) & mw_year_start == 1961 & term == "year0" & abs(estimate) < 0.01]
# 
# dat_lm[!is.na(statistic)] %>% 
#   dcast(Name + month + mw_year_start + mw_year_end ~ term, value.var = "estimate") %>% 
#   .[! (abs(`(Intercept)`) < 1 & abs(year0) < 0.05)] -> dat_lm_sub

# or not
dat_lm[!is.na(statistic)] %>%
  dcast(Name + month + mw_year_start + mw_year_end ~ term, value.var = "estimate") -> dat_lm_sub



# scatter plots ---------------------------------------------------------------

# dat_lm_sub2 <- dat_lm_sub[mw_year_start %in% c(1961, 1966, 1971, 1976, 1981, 1986, 1990)]

dat_plot <- merge(dat_lm_sub, dat_meta_clust, by = "Name")
mitmatmisc::add_month_fct(dat_plot, 10)
dat_plot[, mw_year_fct := paste0(mw_year_start,"-",mw_year_end)]


# adjust ylimits of free panels
dat_blank <- dat_plot[, .(ymin = min(year0), ymax = max(year0)), .(month_fct, mw_year_fct)]
setorder(dat_blank, month_fct, mw_year_fct)
dat_blank %>% 
  melt(measure.vars = c("ymin", "ymax")) %>% 
  dcast(mw_year_fct + variable ~ month_fct) -> dat_blank2
dat_blank2[, Nov := Dec]
dat_blank2[, Jan := Feb]
dat_blank2[, Apr := Mar]
dat_blank2[, May := Mar]
dat_blank3 <- melt(dat_blank2, id.vars = c("mw_year_fct", "variable"), variable.name = "month_fct")


pdf("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/zenodo/aux_paper/Moving window trends all.pdf",
    width = 12, height = 8)

for(i_mw in sort(unique(dat_plot$mw_year_fct))){
  
  gg1 <- dat_plot[mw_year_fct %in% i_mw & month %in% c(11, 12, 1:2)] %>% 
    ggplot(aes(year0*10, Elevation, colour = cluster_fct))+
    geom_vline(xintercept = 0, colour = "grey10")+
    # geom_point(size = 0.4, alpha = 0.3)+
    geom_point(size = 0.3)+
    # geom_smooth(se = F, size = 0.5)+
    scale_color_brewer("Region", palette = "Set1")+
    # scale_x_continuous(n.breaks = 4)+
    facet_grid(. ~ month_fct, scales = "free_x", space = "free_x")+
    geom_blank(inherit.aes = F, data = dat_blank3[month_fct %in% month.abb[c(11, 12, 1:2)]], 
               aes(y = 0, x = value*10))+
    theme_bw(14)+
    theme(panel.grid.minor = element_blank(),
          # panel.spacing.x = unit(12, "pt"),
          legend.position = "right")+
    guides(color = guide_legend(override.aes = list(size = 4)))+
    ylab("Elevation [m]")+
    xlab("Linear trend in mean monthly HS [cm per decade]")
  
  gg2 <- dat_plot[mw_year_fct %in% i_mw & ! month %in% c(11, 12, 1:2)] %>% 
    ggplot(aes(year0*10, Elevation, colour = cluster_fct))+
    geom_vline(xintercept = 0, colour = "grey10")+
    # geom_point(size = 0.4, alpha = 0.3)+
    geom_point(size = 0.3)+
    # geom_smooth(se = F, size = 0.5)+
    scale_color_brewer("Region", palette = "Set1")+
    # scale_x_continuous(n.breaks = 4)+
    facet_grid(. ~ month_fct, scales = "free_x", space = "free_x")+
    geom_blank(inherit.aes = F, data = dat_blank3[! month_fct %in% month.abb[c(11, 12, 1:2)]], 
               aes(y = 0, x = value*10))+
    theme_bw(14)+
    theme(panel.grid.minor = element_blank(),
          # panel.spacing.x = unit(12, "pt"),
          legend.position = "right")+
    guides(color = guide_legend(override.aes = list(size = 4)))+
    ylab("Elevation [m]")+
    xlab("Linear trend in mean monthly HS [cm per decade]")
  
  gg_out <- wrap_plots(gg1, gg2, ncol = 1, guides = "collect")+
    plot_annotation(title = paste0("Trends for the period: ", i_mw))
  
  print(gg_out)
}

dev.off()
