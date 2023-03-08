# plot trends (full long period)


library(data.table)
library(magrittr)
library(lubridate)
library(ggplot2)
library(scico)

library(forcats)
library(foreach)
library(flextable)
library(officer)

pval_sym <- function(pval){
  symnum(pval, corr = FALSE,
         cutpoints = c(0,  .001,.01,.05, .1, 1),
         symbols = c("***","**","*","."," "))
}

dat_meta_clust <- readRDS("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/meta-with-cluster-01.rds")

load("/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/rds/trends-01-1971-2019-ols-gls.rda")


# check if ols or gls -----------------------------------------------------


dat_month_gls[term == "year0"] %>% summary
dat_month_gls[term == "year0", table(pval_sym(p.value.var))]
dat_month_gls[term == "year0", table(pval_sym(p.value.var), month)]
dat_month_gls[term == "year0", table(p.value.var < 0.05, month)] %>% prop.table(2)
dat_month_gls[term == "year0", table(p.value.var < 0.05)] %>% prop.table()

dat_seasonal_gls[term == "year0", table(pval_sym(p.value.var))]
dat_seasonal_gls[term == "year0", table(pval_sym(p.value.var), variable)]
dat_seasonal_gls[term == "year0", table(p.value.var < 0.05, variable)] %>% prop.table(2)
dat_seasonal_gls[term == "year0", 
                 table(pval = p.value.var < 0.05, scd_yn = startsWith(variable, "SCD"))] %>% prop.table(2)

dat_comp_seas_ols_gls <- merge(
  dat_seasonal_gls[term == "year0", 
                   .(Name, variable, est_gls = estimate, pval_gls = p.value)],
  dat_seasonal_ols[term == "year0", 
                   .(Name, variable, est_ols = estimate, pval_ols = p.value)]
)
  
dat_comp_seas_ols_gls %>% with(table(ols = pval_ols < 0.05, gls = pval_gls < 0.05, variable))
dat_comp_seas_ols_gls %>% 
  ggplot(aes(est_ols, est_gls))+
  geom_point()+
  geom_abline()+
  facet_wrap(~variable, scales = "free")

# GLS plot ------------------------------------------------------


dat_plot_full <- dat_month_gls[term == "year0"] %>% 
  merge(dat_meta_clust, by = "Name")
dat_plot_full[, est_low := estimate - 1.96 * std.error]
dat_plot_full[, est_high := estimate + 1.96 * std.error]
mitmatmisc::add_month_fct(dat_plot_full, 10)

dat_plot_full


# manual limits y (elev)
dat_ylim <- dat_plot_full[, .(max_elev = max(Elevation)), .(cluster_fct)] 
setorder(dat_ylim, cluster_fct)
dat_ylim[, max_elev := c(1250, 1250, 3000, 3000, 1250)]


gg <- dat_plot_full %>% 
  ggplot(aes(estimate*10, Elevation, xmin = est_low*10, xmax = est_high*10, colour = cluster_fct))+
  geom_vline(xintercept = 0)+
  geom_pointrange(size = 0.1, fatten = 0.5)+
  # geom_jitter(width = 0, height = 0.3)+
  # facet_wrap(~month_fct, nrow = 2)+
  facet_grid(cluster_fct ~ month_fct, scales = "free", space = "free")+ # free_x or free
  theme_bw()+
  theme(panel.grid.minor = element_blank())+
  scale_color_brewer("", palette = "Set1", guide = F)+
  scale_x_continuous(breaks = 10*seq(-4, 4, by = 2))+
  scale_y_continuous(breaks = seq(0, 3000, by = 500), limits = c(0, NA))+
  geom_blank(inherit.aes = F, data = dat_ylim, aes(x = 0, y = max_elev))+
  xlab("Linear trend in monthly mean HS [cm per decade]")+
  ylab("Elevation [m]")


ggsave(gg,
       filename = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/fig/Figure 5.png",
       width = 10,
       height = 11)







# summary table 6 elev ------------------------------------------------------------

dat_table <- copy(dat_plot_full)


dat_table[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
dat_table[, trend_sign := ifelse(estimate <= 0, "neg", "pos")]

dat_table_avg <- dat_table[,
                           .(nn = as.double(.N),
                             mu = 10*mean(estimate)),
                           .(month_fct, elev_fct, cluster_fct)]


dat_table_sig <- dat_table[, 
                           .(nn = .N), 
                           .(month_fct, elev_fct, cluster_fct, p_sig = p.value < 0.05, trend_sign)]
dat_table_sig[, perc := nn / sum(nn), .(month_fct, elev_fct, cluster_fct)]

dat_table_sig[p_sig == T] %>% 
  dcast(month_fct + elev_fct + cluster_fct ~ trend_sign, value.var = "perc") %>% 
  merge(dat_table_avg, all = T) -> dat_table_out


dat_table_out2 <- dat_table_out %>% 
  melt(id.vars = c("month_fct", "elev_fct", "cluster_fct"))
dat_table_out2[, variable := factor(variable, levels = c("nn", "mu", "neg", "pos"))]
setorder(dat_table_out2, month_fct, elev_fct, cluster_fct, variable)

dat_table_out3 <- dat_table_out2 %>% 
  dcast(month_fct + cluster_fct ~ elev_fct + variable, value.var = c("value"))

dat_header <- data.table(col_keys = names(dat_table_out3))
dat_header[, c("row1", "row2") := tstrsplit(col_keys, "_")]
# dat_header[1:2, ":="(row1 = c("",""), row2 = c("Month", "Region"))]
dat_header[1:2, ":="(row1 = c("Month", "Region"), row2 = c("", ""))]
row2_rename <- setNames(c("#", "mean", "sig-", "sig+"),
                        c("nn", "mu", "neg", "pos"))
dat_header[, row2 := row2_rename[row2]]
dat_header[3:.N, row1 := paste0("Elevation: ", row1, " m")]

cols_perc <- dat_header[row2 %in% c("sig-", "sig+"), col_keys]
dat_table_out3[, c(cols_perc) := lapply(.SD, scales::percent, .1), .SDcols = cols_perc]

dat_table_out3[, cluster_fct := fct_recode(cluster_fct, 
                                           "N&hA" = "North & high Alpine",
                                           "S&hA" = "South & high Alpine")]

ft <- dat_table_out3 %>% 
  flextable() %>%
  set_header_df(dat_header) %>% 
  colformat_int(j = dat_header[row2 == "#", col_keys]) %>% 
  colformat_num(j = dat_header[row2 == "mean", col_keys]) %>% 
  # colformat_num(j = dat_header[row2 == "mu", col_keys]) %>% 
  theme_booktabs() %>%
  hline(i = 1:6*5, border = fp_border()) %>% 
  vline(j = 2 + 0:5*4, border = fp_border()) %>% 
  merge_h(part = "header") %>%
  # align(align = "left", part = "all") %>%
  align(j = dat_header[1:2, col_keys], align = "left") %>% 
  align(j = dat_header[-c(1:2), col_keys], align = "center", part = "all") %>% 
  merge_v(j = "month_fct") %>%
  valign(j = "month_fct", valign = "top") %>%
  fix_border_issues() %>% 
  fontsize(size = 6, part = "all") %>% 
  font(fontname = "Times New Roman", part = "all") # %>% autofit() 

read_docx() %>% 
  body_add_flextable(ft) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/table/supplement/trends-gls-month-500m.docx")




# summary table 3 elev ------------------------------------------------------------

dat_table <- copy(dat_plot_full)


# dat_table[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 500), dig.lab = 5)]
dat_table[, elev_fct := cut(Elevation, breaks = seq(0, 3000, by = 1000), dig.lab = 5)]
dat_table[, trend_sign := ifelse(estimate <= 0, "neg", "pos")]

dat_table_avg <- dat_table[,
                           .(nn = as.double(.N),
                             mu = 10*mean(estimate)),
                           .(month_fct, elev_fct, cluster_fct)]


dat_table_sig <- dat_table[, 
                           .(nn = .N), 
                           .(month_fct, elev_fct, cluster_fct, p_sig = p.value < 0.05, trend_sign)]
dat_table_sig[, perc := nn / sum(nn), .(month_fct, elev_fct, cluster_fct)]

dat_table_sig[p_sig == T] %>% 
  dcast(month_fct + elev_fct + cluster_fct ~ trend_sign, value.var = "perc") %>% 
  merge(dat_table_avg, all = T) -> dat_table_out


dat_table_out2 <- dat_table_out %>% 
  melt(id.vars = c("month_fct", "elev_fct", "cluster_fct"))
dat_table_out2[, variable := factor(variable, levels = c("nn", "mu", "neg", "pos"))]
setorder(dat_table_out2, month_fct, elev_fct, cluster_fct, variable)

dat_table_out3 <- dat_table_out2 %>% 
  dcast(month_fct + cluster_fct ~ elev_fct + variable, value.var = c("value"))


dat_header <- data.table(col_keys = names(dat_table_out3))
dat_header[, c("row1", "row2") := tstrsplit(col_keys, "_")]
# dat_header[1:2, ":="(row1 = c("",""), row2 = c("Month", "Region"))]
dat_header[1:2, ":="(row1 = c("Month", "Region"), row2 = c("", ""))]
row2_rename <- setNames(c("#", "mean", "sig-", "sig+"),
                        c("nn", "mu", "neg", "pos"))
dat_header[, row2 := row2_rename[row2]]
dat_header[3:.N, row1 := paste0("Elevation: ", row1, " m")]

cols_perc <- dat_header[row2 %in% c("sig-", "sig+"), col_keys]
dat_table_out3[, c(cols_perc) := lapply(.SD, scales::percent, .1), .SDcols = cols_perc]

dat_table_out3[, cluster_fct := fct_recode(cluster_fct, 
                                           "N&hA" = "North & high Alpine",
                                           "S&hA" = "South & high Alpine")]

ft <- dat_table_out3 %>% 
  flextable() %>%
  set_header_df(dat_header) %>% 
  colformat_int(j = dat_header[row2 == "#", col_keys]) %>% 
  colformat_num(j = dat_header[row2 == "mean", col_keys]) %>% 
  # colformat_num(j = dat_header[row2 == "mu", col_keys]) %>% 
  theme_booktabs() %>%
  hline(i = 1:6*5, border = fp_border()) %>% 
  vline(j = 2 + 0:2*4, border = fp_border()) %>% 
  merge_h(part = "header") %>%
  # align(align = "left", part = "all") %>%
  align(j = dat_header[1:2, col_keys], align = "left") %>% 
  align(j = dat_header[-c(1:2), col_keys], align = "center", part = "all") %>% 
  merge_v(j = "month_fct") %>%
  valign(j = "month_fct", valign = "top") %>%
  fix_border_issues() %>% 
  fontsize(size = 9, part = "all") %>% 
  font(fontname = "Times New Roman", part = "all") # %>% autofit() 



read_docx() %>% 
  body_add_flextable(ft) %>% 
  print(target = "/mnt/CEPH_PROJECTS/ALPINE_WIDE_SNOW/PAPER/02_review/table/Table 2.docx")


# some numbers ------------------------------------------------------------



with(dat_table, table(trend_sign))
with(dat_table, table(trend_sign)) %>% prop.table()


with(dat_table, table(trend_sign, pval = p.value < 0.05))
with(dat_table, table(trend_sign, pval = p.value < 0.05)) %>% prop.table()

dat_table[, 10*mean(estimate), keyby = .(month_fct, elev_fct)]

dat_table[month %in% c(12, 1, 2), 
          .(.N, 10*mean(estimate), 10*min(estimate), 10*max(estimate)), 
            keyby = .(elev_fct)]
dat_table[month %in% c(3:5), 
          .(.N, 10*mean(estimate), 10*min(estimate), 10*max(estimate)), 
          keyby = .(elev_fct)]

