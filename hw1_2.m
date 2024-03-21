clear all

% 一维边值问题
N = 32; % 剖分
h = 1 / N;
K = sparse(N-1, N-1); % 初始化刚度矩阵
f = @(x) (pi^2 + 1) * sin(pi * x);

for i = 1 : N-1
    K(i, i) = 1/h + 1/h + (h+h)/3;

    F(i) = (h/2 + h/2) * f(i*h); % 载荷向量
end

for i = 1 : N-2
    K(i, i+1) = -1/h + h/6;
end

for i = 2 : N-1
    K(i, i-1) =  -1/h + h/6;
end

F = F';

U = K \ F;

x = linspace(0, 1, 101); 
uh = zeros(1, length(x));
u = sin(pi * x);

for j = 1: length(x)
    for i = 1 : N-1 
        uh(j) = uh(j) + U(i) * phi(x(j), i, N);
    end
end

err = abs(u - uh);

plot(x, uh)

function s = phi(x, i, N) % 分段插值线性函数
   
    if x >= (i-1)*(1/N) && x <= i*(1/N)
        s = N * (x - (i-1)*(1/N));
    elseif x >= i*(1/N) && x <= (i+1)*(1/N)
        s = N * ((i+1)*(1/N) - x);
    else
        s = 0;
    end
    
end

