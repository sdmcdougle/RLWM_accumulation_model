%% Model Simulation code for the piH model
%% SD McDougle & AGE Collins; Psychonomic Bulletin & Review; (2020)

% this MATLAB script simulates RTs and choices using an optimized piH model
% with params gleaned from fitting the model to these data (using MLE)
% outputs of these simulations can be used to generate all figures related
% to these data (Figures 1-8)



clear all;clc;close all; % init

load piH_model; % load optimized model of interest
mod = fit_object; % re-name

data = 'dataSet1'; %
load([data,'.mat']); % load data
num_subs = size(data,2); % number of participants
num_sims = 5; % N simulations per subject's param values (increase (~100) for final analyses

na = 3; % number of acvailable actions in the task

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SIM FULL RT / CHOICE MODEL %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for si = 1:num_subs
    
    disp(['simulating subject ',' ',num2str(si)]); % track progress
    
    %% specify trial blocks
    blocks = data(si).block_data{end}.blocks;
    
    %% specify model params
    alpha = mod.alpha(si); % RL learning rate
    alpha_neg = mod.alpha_neg(si); % perseveration on negative outcomes
    phi = mod.phi(si); % working memory decay
    rho = mod.rho(si); % working memory weighting
    C = mod.C(si); % working memory capacity
    eta = mod.eta(si); % drift rate scale factor
    A = mod.A(si); % initial bias
    bound = mod.bound(si); % boundary/threshold
    s_v = mod.s_v(si); % drift noise (fixed)
    t_0 = mod.t_0(si); % non-decision time (fixed)
    beta = mod.beta(si); % inverse temperature (fixed)
    
    %% loop over simulations %%
    for sims = 1:num_sims
        
        %% initializing/storing multiple variables of interest for clarity/plotting %%
        % init
        perf = nan(length(blocks),15); % performance of model
        sub_perf = nan(length(blocks),15); % performance of subject
        rt_tc = nan(length(blocks),15); % model rt time course
        sub_rt_tc = nan(length(blocks),15); % subject rt time course
        iter = nan(length(blocks),90); % iteration of current stimulus
        setsize = nan(length(blocks),90); % set size
        rttmp = nan(length(blocks),90); % rt sub
        RTtmp = nan(length(blocks),90); % rt model
        
        SUBCOR = nan(length(blocks),90);
        SIMCOR = nan(length(blocks),90);
        
        %% loop over trial block %%
        for b = 1:length(blocks)
            ns(b)=blocks(b); % set size per block
            bdata = data(si).block_data{b}; % data (if desired)
            reward = bdata.Cor; % reward
            reward(reward<0) = NaN; % screen error trials (coded as "-1")
            num_trials = length(reward); % trials performed in block
            seq = bdata.seq(1:num_trials); % stimulus sequence
            rt = bdata.RT*1000; % convert s to ms if desired
            rt(rt<150) = NaN; % screen slip-up trials
            rttmp(b,1:num_trials) = rt; % store if desired
            sub_action = bdata.Code; % subject action
            cor_action = bdata.actionseq(1:num_trials); % correct action
            
            % init behavioral variables
            cor_stim = nan(ns(b),15); % correct (by iteration)
            rt_stim = nan(ns(b),15); % rt (by iteration)
            sub_cor_stim = nan(ns(b),15); % sub correct (by iteration)
            sub_rt_stim = nan(ns(b),15); % sub rt (by iteration)
            s_p = [];RT = [];r = []; % init
            
            % init model latents
            q_rl = ones(ns(b),na)*(1/na); % RL q values
            q_wm = ones(ns(b),na)*(1/na); % WM "q" values
            weight = rho * min(1,C/ns(b)); % initial weighting of WM
            
            %% trial loop
            for i = 1:num_trials
                
                % store
                setsize(b,i) = ns(b);
                
                s = seq(i); % stimulus on this trial
                a = sub_action(i); % action taken on this trial
                
                % stor sub correct
                SUBCOR(b,i) = reward(i);
                
                % RL policy pi
                p_rl = mcdougle_softmax_func(q_rl(s,:),beta);
                % WM policy pi
                p_wm = mcdougle_softmax_func(q_wm(s,:),beta);
                
                % weighted combined choice policy
                pol = (1-weight)*p_rl + weight*p_wm; % policy vector
                
                %% compute policy entropy on state-averaged policy %%
                % first extract policy for each state/stimulus
                for kk = 1:ns(b)
                    % RL pol
                    temp_p_rl = mcdougle_softmax_func(q_rl(kk,:),beta);
                    % WM pol
                    temp_p_wm = mcdougle_softmax_func(q_wm(kk,:),beta);
                    % weighted
                    s_p(kk,:) = (1-weight)*temp_p_rl + weight*temp_p_wm; % policy vector
                end
                % compute "prior" policy
                if ns(b) > 1
                    mean_p = mean(s_p);
                else
                    mean_p = s_p;
                end
                % compute prior entropy
                prior_ent = -sum(mean_p.*log2(mean_p));
                
                %% piH model drift computation %%
                v = eta .* (pol./prior_ent);
                
                %% "pi" model drift computation
                % v = eta .* pol;
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% RT and choice simulation %%
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                trialOK = false; % valid sim flag
                while ~trialOK
                    for act = 1:na
                        % Get starting point
                        k(act) = rand.*A;
                        
                        % Get drift rate
                        d(act) = normrnd(v(act), s_v);
                        
                        % Get time to threshold
                        t(act) = (bound-k(act))/d(act);
                        
                        % Add on non-decision time
                        allRT(act) = t_0 + t(act);
                    end
                    
                    % extract choice and RT
                    allRT(d<0) = nan; % exclude negative drifts
                    [RT(i), sim_a] = nanmin(allRT);
                    % re-run sim for invalid RTs
                    if RT(i) < 1400 % 1400 max RT (for this experiment; Collins and Frank 2018)
                        trialOK = true;
                    end
                    
                end
                
                % was model correct?
                if sim_a == cor_action(i)
                    r(i) = 1;
                else
                    r(i) = 0;
                end
                % store
                SIMCOR(b,i) = r(i);
                
                %% Q-learning for RL and WM modules %%
                if r(i) == 1 % pos rpe
                    q_rl(s,sim_a) = q_rl(s,sim_a) + alpha*(r(i)-q_rl(s,sim_a));
                    q_wm(s,sim_a) = q_wm(s,sim_a) + 1*(r(i)-q_wm(s,sim_a)); % perfect learning rate for WM
                else % neg rpe
                    q_rl(s,sim_a) = q_rl(s,sim_a) + alpha_neg*alpha*(r(i)-q_rl(s,sim_a));
                    q_wm(s,sim_a) = q_wm(s,sim_a) + alpha_neg*1*(r(i)-q_wm(s,sim_a));
                end
                
                % working memory forgetting/decay
                q_wm = q_wm + phi*((1/na)-q_wm);
                
            end
            RTtmp(b,1:num_trials) = RT; % store
            
            %% below is code for setting up variables for paper's time course plots %%
            for k = 1:ns(b)
                cor_stim(k,1:length(r(seq == k))) = r(seq == k);
                sub_cor_stim(k,1:length(r(seq == k))) = reward(seq==k);
                rt_stim(k,1:length(r(seq == k))) = RT(seq == k);
                sub_rt_stim(k,1:length(r(seq == k))) = rt(seq==k);
            end
            if ns(b) > 1
                perf(b,:) = nanmean(cor_stim);
                sub_perf(b,:) = nanmean(sub_cor_stim);
                rt_tc(b,:) = nanmean(rt_stim);
                sub_rt_tc(b,:) = nanmean(sub_rt_stim);
            else
                perf(b,:) = cor_stim;
                sub_perf(b,:) = sub_cor_stim;
                rt_tc(b,:) = rt_stim;
                sub_rt_tc(b,:) = sub_rt_stim;
            end
            
        end
        
        %% flattened data (also for plots and stuff)
        flatsetsize = setsize(:);flatiter = iter(:);
        flatsub = rttmp(:);flatsim = RTtmp(:);
        flatsimcor = SIMCOR(:);flatsubcor = SUBCOR(:);
        
        %% quantiles (plots)
        NQ = 5;
        for jj = 1:6
            sub_quant{jj}(si,:) = quantile(flatsub(flatsetsize==jj),NQ);
            sim_quant_tmp{jj}(sims,:) = quantile(flatsim(flatsetsize==jj),NQ);
            sub_quant_cor{jj}(si,:) = quantile(flatsub(flatsetsize==jj & flatsubcor==1),NQ);
            sim_quant_tmp_cor{jj}(sims,:) = quantile(flatsim(flatsetsize==jj & flatsimcor==1),NQ);
            sub_quant_incor{jj}(si,:) = quantile(flatsub(flatsetsize==jj & flatsubcor==0 & flatiter>2),NQ);
            sim_quant_tmp_incor{jj}(sims,:) = quantile(flatsim(flatsetsize==jj & flatsimcor==0 & flatiter>2),NQ);
        end
        
        %% store info for each sim
        for gi = 1:max(ns)
            sim_perf{gi}(sims,:) = nanmean(perf(ns==gi,1:9)); % sim p cor
            sub_perf_all{gi}(1,:) = nanmean(sub_perf(ns==gi,1:9)); % sub p cor
            sim_rttc{gi}(sims,:) = nanmean(rt_tc(ns==gi,1:9)); % sim rt
            sub_rttc_all{gi}(1,:) = nanmean(sub_rt_tc(ns==gi,1:9)); % sub rt
        end
        
    end
    
    %% sim quantile avg over sims
    for jj = 1:6
        sim_quant{jj}(si,:) = nanmean(sim_quant_tmp{jj});
        sim_quant_cor{jj}(si,:) = nanmean(sim_quant_tmp_cor{jj});
        sim_quant_incor{jj}(si,:) = nanmean(sim_quant_tmp_incor{jj});
    end
    
    % means over sims
    for ii = 1:max(ns)
        NS{ii}(si,:) = nanmean(sim_perf{ii});
        sub_ns{ii}(si,:) = sub_perf_all{ii};
        TC{ii}(si,:) = nanmean(sim_rttc{ii});
        sub_tc{ii}(si,:) = sub_rttc_all{ii};
    end
    
end

%%%%%%%%%%%
%% PLOTS %%
%%%%%%%%%%%

%% choice and RT time course figures
figure;
cs = {'k',[.25 .38 .85],[.26 .85 .96],[.24 .72 .3],[.9 .9 .1],[.9 .1 .3]};

% choice
subplot(1,2,1); % TCs
plot(0,0,'color',cs{1},'linewidth',2);hold on;plot(0,0,'color',cs{1},'linewidth',2);
for j = 1:max(ns)
    shadedErrorBar(1:9,nanmean(sub_tc{j}),nanstd(sub_tc{j})/sqrt(num_subs)*1.96,{'color',cs{j}},.1);hold on;
end
for j = 1:max(ns)
    plot(1:9,nanmean(sub_tc{j}),'color',cs{j},'linewidth',2);hold on;
    plot(1:9,nanmean(TC{j}),'--','color',cs{j},'linewidth',2);
end

xlabel('Stimulus iteration');ylabel('RT (ms)')
axis([0 11.5 50 800]);
set(gca,'xtick',1:9);
fs = 8;
text(9.5,260,'nS1','color',cs{1},'fontsize',fs);
text(9.5,410,'nS2','color',cs{2},'fontsize',fs);
text(9.5,500,'nS3','color',cs{3},'fontsize',fs);
text(9.5,540,'nS4','color',cs{4},'fontsize',fs);
text(9.5,580,'nS5','color',cs{5},'fontsize',fs);
text(9.5,640,'nS6','color',cs{6},'fontsize',fs);
legend('data','\pi_{H}','location','southwest');
legend('boxoff');
box off;

%% RT
subplot(1,2,2);
for j = 1:max(ns)
    shadedErrorBar(1:9,nanmean(sub_ns{j}),nanstd(sub_ns{j})/sqrt(num_subs)*1.96,{'color',cs{j}},.1);hold on;
end
for j = 1:max(ns)
    plot(1:9,nanmean(sub_ns{j}),'color',cs{j},'linewidth',2);hold on;
    plot(1:9,nanmean(NS{j}),'--','color',cs{j},'linewidth',2);
end
xlabel('Stimulus iteration');ylabel('p(correct)')
set(gca,'xtick',1:9);
box off;

set(gcf,'Position',[1 569 426 236]);



%% QUANITLE plots
figure;hold on;
for i = 1:6
    plot(nanmean(sub_quant{i}),'color',cs{i});
    plot(nanmean(sim_quant{i}),'--','color',cs{i});
    for ki = 1:5
        SDM_errorbars(ki,nanmean(sub_quant{i}(:,ki)),nanstd(sub_quant{i}(:,ki))/sqrt(num_subs)*1.96,cs{i},1,.05);
    end
end
fs = 10;xx = 5.15;
text(xx,410,'nS1','color',cs{1},'fontsize',fs);
text(xx,570,'nS2','color',cs{2},'fontsize',fs);
text(xx,680,'nS3','color',cs{3},'fontsize',fs);
text(xx,770,'nS4','color',cs{4},'fontsize',fs);
text(xx,820,'nS5','color',cs{5},'fontsize',fs);
text(xx,885,'nS6','color',cs{6},'fontsize',fs);
legend('data','\pi_{H}','location','northwest');
legend('boxoff');axis([.5 5.5 150 1000]);
box off;
set(gca,'xtick',1:5,'xticklabel',{'.17','.33','.50','.67','.83'});
xlabel('Quantile');
ylabel('RT (ms)');
set(gca,'fontsize',12);
set(gcf,'Position',[1 169 460 326]);






