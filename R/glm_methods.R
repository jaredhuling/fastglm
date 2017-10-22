
#' print method for fastglm objects
#'
#' @param x object to print
#' @param ... not used
#' @rdname print
#' @method print fastglm
#' @export
print.fastglm <- function(x, ...) {
    cat("\nCall:\n")
    print(x$call)
    cat("\nCoefficients:\n")
    print(x$coefficients, digits=5)
}
