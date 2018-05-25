function [g, g_dd] = compute_g(t)

t2 = ceil(5*sqrt(t));

x = -5*sqrt(t) : 5*sqrt(t);

g = 1/sqrt(2*pi*t) * exp(-x.^2 /(2*t));

g_dd = (1/sqrt(2*pi*t) * exp(-x.^2/(2*t)) .* (x.^2 - t)) / t^2;

end

