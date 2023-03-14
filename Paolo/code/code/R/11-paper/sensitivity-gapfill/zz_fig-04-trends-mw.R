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

dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/sensitivity-gapfill/meta-with-cluster-01.rds")

# choose hydro_year or calendar year
# dat_lm <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/sensitivity-gapfill/trends-01-mw-hydro-year.rds")
dat_lm  <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/rds/sensitivity-gapfill/trends-02-mw-calendar-year.rds")

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

dat_lm_sub2 <- dat_lm_sub[mw_year_start %in% c(1961, 1966, 1971, 1976, 1981, 1986)]

dat_plot <- merge(dat_lm_sub2, dat_meta_clust, by = "Name")
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

# gg <- dat_plot %>% 
#   ggplot(aes(Elevation, year0, colour = cluster_fct))+
#   geom_hline(yintercept = 0, colour = "grey10")+
#   # geom_point(size = 0.4, alpha = 0.3)+
#   geom_point(size = 0.1)+
#   # geom_smooth(se = F, size = 0.5)+
#   scale_color_brewer("Region", palette = "Set1")+
#   facet_grid(month_fct ~ mw_year_fct, scales = "free_y", space = "free_y")+
#   geom_blank(inherit.aes = F, data = dat_blank3, aes(x = 0, y = value))+
#   theme_bw()+
#   theme(panel.grid.minor = element_blank())+
#   xlab("Elevation [m]")+
#   ylab("Linear trend in mean monthly HS [cm/year]")

gg <- dat_plot %>% 
  ggplot(aes(year0*10, Elevation, colour = cluster_fct))+
  geom_vline(xintercept = 0, colour = "grey10")+
  # geom_point(size = 0.4, alpha = 0.3)+
  geom_point(size = 0.3)+
  # geom_smooth(se = F, size = 0.5)+
  scale_color_brewer("Region", palette = "Set1")+
  scale_x_continuous(n.breaks = 4)+
  facet_grid(mw_year_fct ~ month_fct, scales = "free_x", space = "free_x")+
  geom_blank(inherit.aes = F, data = dat_blank3, aes(y = 0, x = value*10))+
  theme_bw(14)+
  theme(panel.grid.minor = element_blank(),
        panel.spacing.x = unit(12, "pt"))+
  guides(color = guide_legend(override.aes = list(size = 4)))+
  ylab("Elevation [m]")+
  xlab("Linear trend in mean monthly HS [cm per decade]")



ggsave(gg,
       file = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/fig/sensitivity-gapfill/Figure 5.png",
       width = 12, height = 12)  

# supplement: table with constant station set? needed?
# not really, there is afterwards the complete period stuff!
dat_lm[term == "year0", .N, .(month, mw_year_start)] %>% 
  ggplot(aes(mw_year_start, N))+
  geom_point()+
  facet_wrap(~month)



# table of histogram (needed?) ------------------------------------------------------
# 
# dat_table <- copy(dat_plot)
# 
# dat_table$estimate %>% qplot
# dat_table$estimate %>% summary
# dat_table[, trend_bins := cut(estimate, breaks = -10 : 6)]
# dat_table[, .N, keyby = trend_bins]
# 
# dat_table[, trend_bins := cut(estimate, breaks = c(-10, -4:4, 6), right = F)]
# 
# dat_table2 <- dat_table[, .(nn = .N), .(trend_bins, month_fct, mw_year_fct)]
# dat_table2[, perc := nn/sum(nn), .(month_fct, mw_year_fct)]
# 
# dat_table3 <- dat_table2 %>% 
#   dcast(month_fct + trend_bins ~ mw_year_fct, value.var = list("nn", "perc"))
# 
# # dat_table3[, trend_bins := as.character(trend_bins)]
# setorder(dat_table3, month_fct, -trend_bins)
# 
# dat_header <- data.table(col_keys = colnames(dat_table3))
# dat_header[, header2 := tstrsplit(col_keys, "_", keep = 1)]
# dat_header[header2 == "nn", header2 := "# count"]
# dat_header[header2 == "perc", header2 := "Percent"]
# dat_header[c(1,2) , header2 := ""]
# dat_header[, header := tstrsplit(col_keys, "_", keep = 2)]
# dat_header[c(1,2), header := c("Month", "Trend [cm/yr]")]
# 
# 
# dat_table3 %>% 
#   flextable() %>% 
#   set_header_df(dat_header) %>% 
#   merge_h(part = "header") %>% 
#   theme_booktabs() %>% 
#   colformat_int(na_str = "",
#                 j = c("nn_1961-1990", "nn_1971-2000", "nn_1981-2010", "nn_1990-2019")) %>% 
#   set_formatter(`perc_1961-1990` = scales::percent_format(1),
#                 `perc_1971-2000` = scales::percent_format(1),
#                 `perc_1981-2010` = scales::percent_format(1),
#                 `perc_1990-2019` = scales::percent_format(1)) %>%
#   merge_v() %>% 
#   valign(valign = "top") %>% 
#   fix_border_issues() %>% 
#   autofit() 
