# Figures for the ordinal comparison, shared by the benchmark script and vignette.
# Each method keeps one color and shape everywhere; fastglm is the dominant line.
ordinal_method_order <- c("fastglm", "rms::orm", "ordinal::clm", "MASS::polr", "VGAM::vglm",
                          "WeightIt::ordinal_weightit")
ordinal_method_style <- function(plot) {
    plot +
        ggplot2::scale_colour_manual(values = c("#0B5CAD", "#D1661F", "#1F9E8F", "#9A62C4", "#9C8200", "#D0628E"),
                                     limits = ordinal_method_order, drop = FALSE) +
        ggplot2::scale_shape_manual(values = c(16, 17, 15, 18, 3, 4), limits = ordinal_method_order, drop = FALSE) +
        ggplot2::scale_linewidth_manual(values = c(1.2, rep(0.5, 5L)), limits = ordinal_method_order, drop = FALSE)
}
ordinal_theme <- function() {
    ggplot2::theme_bw(base_size = 12) +
        ggplot2::theme(panel.grid.minor = ggplot2::element_blank(), legend.position = "bottom",
                       panel.spacing.x = ggplot2::unit(1.1, "lines"), strip.background = ggplot2::element_blank(),
                       strip.text = ggplot2::element_text(hjust = 0))
}
ordinal_log_breaks <- 10^(-3:3)
ordinal_log_labels <- c("0.001", "0.01", "0.1", "1", "10", "100", "1,000")

# A fit that disagrees with fastglm has no comparable time, so it is omitted.
plot_ordinal_times <- function(summary) {
    timing <- summary[summary$agrees & summary$K == 5L, ]
    timing$facet <- ifelse(timing$panel == "Common slopes", paste0("Common slopes, ", timing$link),
                           "Threshold-specific slopes, logit")
    timing$facet <- factor(timing$facet, levels = c("Common slopes, logit", "Common slopes, probit",
        "Common slopes, cloglog", "Threshold-specific slopes, logit"))
    timing$method <- factor(timing$method, levels = ordinal_method_order)
    plot <- ggplot2::ggplot(timing, ggplot2::aes(.data$n, .data$median_seconds, colour = .data$method,
            shape = .data$method, linewidth = .data$method, group = .data$method)) +
        ggplot2::geom_line() + ggplot2::geom_point(size = 2.4) +
        ggplot2::scale_x_log10(breaks = c(1000, 10000, 100000), labels = c("1,000", "10,000", "100,000")) +
        ggplot2::scale_y_log10(breaks = ordinal_log_breaks, labels = ordinal_log_labels) +
        ggplot2::facet_wrap(~facet, ncol = 2L) +
        ggplot2::labs(x = "Number of observations", y = "Median seconds per fit, log scale",
                      colour = NULL, shape = NULL, linewidth = NULL)
    ordinal_method_style(plot) + ordinal_theme()
}

plot_ordinal_categories <- function(summary) {
    timing <- summary[summary$panel == "Many categories" & summary$agrees, ]
    timing$method <- factor(timing$method, levels = ordinal_method_order)
    plot <- ggplot2::ggplot(timing, ggplot2::aes(.data$K, .data$median_seconds, colour = .data$method,
            shape = .data$method, linewidth = .data$method, group = .data$method)) +
        ggplot2::geom_line() + ggplot2::geom_point(size = 2.4) +
        ggplot2::scale_x_log10(breaks = c(30, 300, 1000), labels = c("30", "300", "1,000")) +
        ggplot2::scale_y_log10(breaks = ordinal_log_breaks, labels = ordinal_log_labels) +
        ggplot2::labs(x = "Number of outcome categories, log scale", y = "Median seconds per fit, log scale",
                      colour = NULL, shape = NULL, linewidth = NULL)
    ordinal_method_style(plot) + ordinal_theme()
}

# Dense and streamed fits share a color by estimator and differ by linetype.
plot_ordinal_internal <- function(internal) {
    internal$estimator <- ifelse(grepl("bias", internal$way), "Mean bias reduction", "Maximum likelihood")
    internal$storage <- ifelse(grepl("^Dense", internal$way), "Dense", "Streamed")
    ggplot2::ggplot(internal, ggplot2::aes(.data$n, .data$median_seconds, colour = .data$estimator,
            linetype = .data$storage, shape = .data$storage, group = .data$way)) +
        ggplot2::geom_line(linewidth = 0.9) + ggplot2::geom_point(size = 2.4) +
        ggplot2::scale_x_log10(breaks = c(100000, 1000000), labels = c("100,000", "1,000,000")) +
        ggplot2::scale_y_log10(breaks = ordinal_log_breaks, labels = ordinal_log_labels) +
        ggplot2::scale_colour_manual(values = c("#0B5CAD", "#D1661F")) +
        ggplot2::labs(x = "Number of observations", y = "Median seconds per fit, log scale",
                      colour = NULL, linetype = NULL, shape = NULL) +
        ordinal_theme()
}
