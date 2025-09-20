% function M2tool()
% version 1.5
% tino.lang@desy.de
% beam caustic measurement, M2 fitting and peak intensity enhancement calc. for different tigger delays



close all force
clear all

config.doocsAdd_cam   = 'FLASH.LASER/FLASHSLASH1.CAM/SLASH.52.RD1/';
%config.doocsAdd_cam   = 'FLASH.LASER/SLASH1.CAM/FEInc.23.In.Position/'; %use only for testing the scirpt

%% pixel size of the camera
config.pixelSize=15.e-6; 

%% set wavelength for M2 caustic fit
config.wavelength=1030e-9; 

%% aperture / diameter factor for sigma calculation
config.aperture2diameter=3;

%% background substruction methode
config.BGsub = 'none'; %'none';'delay' uses bg image taken for from burst timing; 'blocked' uses bg image when beam is blocked

%% principal axes rotation
config.principalAxesRot = 1; %1 - rotate profile by azimuthal angle accoriding to IS 11146-1
%%

config.N_pointsInBurst = 1;
config.DelaySpan = 0;
config.centerBurstDelay      = 5900;
config.delays                = linspace(config.centerBurstDelay-config.DelaySpan/2, config.centerBurstDelay+config.DelaySpan/2,config.N_pointsInBurst); 
config.doocsAdd_trigDelay    = [config.doocsAdd_cam ,'TRIGGERDELAYABS'];
config.BGdelay               = 10000;
config.NumIMGs4average       = 1;




%% main UI
%%
close all
f_mainUI = uifigure;

f_mainUI.UserData.config    = config;
f_mainUI.UserData.profiles  = [];
f_mainUI.UserData.results   = [];

btn_h  = 20;
n_btns = 7;
set(f_mainUI,'Position',[f_mainUI.Position(1:2) 220 n_btns*btn_h],'Name','M2tool')
%%right side
i_btn=0;

i_btn=i_btn+1;
btn(i_btn) = uibutton(f_mainUI,'push','Text','Linear fits',...
               'Position',[1, btn_h*(i_btn-1), 220, btn_h],...
               'ButtonPushedFcn', @(btn,event) UI_linfit(f_mainUI));
           
i_btn=i_btn+1;
btn(i_btn) = uibutton(f_mainUI,'push','Text','M2 fits',...
               'Position',[1, btn_h*(i_btn-1), 220, btn_h],...
               'ButtonPushedFcn', @(btn,event) UI_M2fit(f_mainUI));

i_btn=i_btn+1;
btn(i_btn) = uibutton(f_mainUI,'push','Text','Take IMGs',...
               'Position',[1, btn_h*(i_btn-1), 220, btn_h],...
               'ButtonPushedFcn', @(btn,event) UI_takeIMG(f_mainUI));

i_btn=i_btn+1;
btn(i_btn) = uibutton(f_mainUI,'push','Text','Get beam radii',...
               'Position',[1, btn_h*(i_btn-1), 220, btn_h],...
               'ButtonPushedFcn', @(btn,event) UI_recalcBeamProbs(f_mainUI)); 
           
i_btn=i_btn+1;
btn(i_btn) = uibutton(f_mainUI,'push','Text','Config',...
               'Position',[1, btn_h*(i_btn-1), 220, btn_h],...
               'ButtonPushedFcn', @(btn,event) UI_config(f_mainUI));               
           
           
i_btn=i_btn+1;
btn(i_btn) = uibutton(f_mainUI,'push','Text','Save images',...
               'Position',[1, btn_h*(i_btn-1), 220, btn_h],...
               'ButtonPushedFcn', @(btn,event) UI_saveIMGs(f_mainUI));              

i_btn=i_btn+1;
btn(i_btn) = uibutton(f_mainUI,'push','Text','Load images',...
               'Position',[1, btn_h*(i_btn-1), 220, btn_h],...
               'ButtonPushedFcn', @(btn,event) UI_loadIMGs(f_mainUI));    
           
i_btn=i_btn+1;
btn(i_btn) = uibutton(f_mainUI,'push','Text','Save project',...
               'Position',[1, btn_h*(i_btn-1), 220, btn_h],...
               'ButtonPushedFcn', @(btn,event) UI_saveProject(f_mainUI));  
           
i_btn=i_btn+1;
btn(i_btn) = uibutton(f_mainUI,'push','Text','Load project',...
               'Position',[1, btn_h*(i_btn-1), 220, btn_h],...
               'ButtonPushedFcn', @(btn,event) UI_load(f_mainUI));             
%%

% load workspace4


% profiles=beamData(profiles,config);
% results = extractBeamProp(profiles,config);
% results = getM2Data(results,config);
%%

function UI_load(f_mainUI)

    [baseName, folder] = uigetfile('.mat','load project');   if baseName==0; return , end
    
    fullpath = fullfile(folder, baseName);
    
    waitMsg(1);
    load(fullpath);    
    waitMsg(0);
    
    if ~exist('project'), msgbox([fullpath,' is no M2tool project file']),end
    
    f_mainUI.UserData = project;
    
    beamData(f_mainUI);
    
end


function UI_saveIMGs(f_mainUI)

    folder = uigetdir(pwd,'choose directory');   if folder==0; return , end
    
    config = f_mainUI.UserData.config; 
    profiles = f_mainUI.UserData.profiles; 
    positions = fieldnames(profiles);
    
    save(fullfile(folder,'config.mat'),'config','-v7.3','-nocompression');
    h=waitbar(0,'save IMGs');
    ht = get(get(h, 'CurrentAxes'), 'title');
    set(ht, 'Interpreter', 'none');
    for i_pos = 1 : length(positions)
        waitbar(i_pos/length(positions),h,['save image data for ',positions{i_pos}]);
        file=fullfile(folder,[positions{i_pos},'.mat']);
        if ~exist(file)
            data = profiles.(positions{i_pos});
            save(file,'data','-v7.3','-nocompression');
        end
    end
    close(h)

    %%

end

function UI_loadIMGs(f_mainUI)

    [files, folder] = uigetfile('*.*','load images','MultiSelect','on');   if ~iscell(files) && ~ischar(files); return , end
    
   
    h=waitbar(0,'load IMGs');
    ht = get(get(h, 'CurrentAxes'), 'title');
    set(ht, 'Interpreter', 'none');
    for N_files = 1 : length(files)
        waitbar(N_files/length(files),h,['load ',files{N_files}]);
        file=fullfile(folder,files{N_files});
        
        if strcmp(files{N_files}(end-3:end),'.mat')
            if strfind(files{N_files},'config')
                load(file);
                if exist('config') == 1
                    f_mainUI.UserData.config = config;
                end
            else
                pos = str2double( regexprep( files{N_files}, {'\D*([\d\.]+\d)[^\d]*', '[^\d\.]*'}, {'$1 ', ' '} ) );
                if isnan(pos)
                    pos = N_files;
                end
                
                if strcmp(files{N_files}(end-3:end),'.mat')
                    data=load(file);
                    if isstruct(data)
                        f_mainUI.UserData.profiles.(['zpos_',num2str(pos)])=data.data;
                    else
                        f_mainUI.UserData.profiles.(['zpos_',num2str(pos)]).IMGraw.trigDelay_0=data;
                    end
                
                end
            end
        else
            pos =  regexprep( files{N_files}, {'\D*([\d\.]+\d)[^\d]*', '[^\d\.]*'}, {'$1 ', ' '} ) ;
            pos(strfind(pos,'.'))='p';
            pos(strfind(pos,' '))=[];
            if isnan(pos)
                pos = N_files;
            end
            try
                IMG = imread(file);                
            catch
                IMG = importdata(file);
            end
            if length(size(IMG))==3
                IMG = sum(IMG,3);
            end
            f_mainUI.UserData.profiles.(['zpos_',pos]).IMGraw.trigDelay_0=IMG; 
       end
        
        
    end
    close(h)

    %%

end



function UI_saveProject(f_mainUI)

    [baseName, folder] = uiputfile('.mat','load project');   if baseName==0; return , end
    
    fullpath = fullfile(folder, baseName);
    
    project = f_mainUI.UserData; 
    
    waitMsg(1);
        save(fullpath,'project','-v7.3','-nocompression'); 
    waitMsg(0);

    %%

end

function waitMsg(state)
    
    persistent f

    if state
        f = figure;
        set(f,'Position',[f.Position(1:2) 200 30],'MenuBar','none','Name','saveing...')
        b = uicontrol(f,'Style','text','String','please wait...','Position',[1 1 200 20]);
        drawnow;
    else
        close(f)
        msgbox('done')
    end
end

function UI_config(f_mainUI)
    
    parameters = {'doocsAdd_cam (use base address of camera server - takes raw image)',...
                  'doocsAdd_trigDelay (from camera server or X2 timer)',...
                  'pixelSize (??m)',...
                  'wavelength (m)',...
                  'aperture2diameter (Factor integration range for 2nd mom. - ISO = 3)',...
                  'BGdelay (??s - takes image of burst timing for background substruction)',...
                  'centerBurstDelay (??s - check this with camera server doocs panel)',...
                  'DelaySpan (??s)',...
                  'N_pointsInBurst (for in-burst dynamics)',...
                  'NumIMGs4average (sums up N IMGs (also BG) according to ISO > 5)',...
                  'BGsub ([none]; [delay] background IMG outside burst; [blocked] BG img with blocked beam',...
                  'principalAxesRot (0 - off; 1 - rotates beam profile by azimuthal angle accoriding to IS 11146-1)',...
                 };

    for i_param =1 :length(parameters)
        tmp = parameters{i_param};  tmp(find(tmp == ' ',1) : end) = [];
        if isfield(f_mainUI.UserData.config,tmp)
            inputs{i_param} = num2str(f_mainUI.UserData.config.(tmp));
        else
            inputs{i_param} = '1';
        end
    end

    answer = inputdlg(parameters,'Config',[1 100],inputs);
    
    if isempty(answer), return, end
    
    for i_param =1 :length(parameters)
        value = str2double(answer(i_param));
        tmp = parameters{i_param};  tmp(find(tmp == ' ',1) : end) = [];
        if isnan(value)
            f_mainUI.UserData.config.(tmp) = answer{i_param};
        else
            f_mainUI.UserData.config.(tmp) = value;
        end
    end 
    beamData(f_mainUI)
end

function UI_takeIMG(f_mainUI)
    takeImages(f_mainUI);
end


function UI_recalcBeamProbs(f_mainUI)
    
    posFields = fields(f_mainUI.UserData.profiles);
    for i_pos=1:length(posFields)
        if isfield(f_mainUI.UserData.profiles.(posFields{i_pos}),'checkSums')
        delayFields = fields(f_mainUI.UserData.profiles.(posFields{i_pos}).checkSums);
        for i_delay=1:length(delayFields)
           f_mainUI.UserData.profiles.(posFields{i_pos}).checkSums.(delayFields{i_delay})=rand;
        end
        end
    end
    
    beamData(f_mainUI); 
    extractBeamProp(f_mainUI);
    
end

function UI_M2fit(f_mainUI)

    beamData(f_mainUI); 
    extractBeamProp(f_mainUI);
    getM2Data(f_mainUI);
    
end

function takeImages(f_mainUI)
%%
    
    config=f_mainUI.UserData.config;
    
    [~] = doocswrite(config.doocsAdd_trigDelay,config.centerBurstDelay);
    z_pos = inputDLG('z_position',{'enter z position (integer in mm)'});
    
    while ~isempty(z_pos)
        
        

        BGbeamblocked=0;
        if strcmp(config.BGsub,'blocked')
            waitfor(msgbox('insert beam block'));
            for i_avr=1:config.NumIMGs4average
                pull = doocsread([config.doocsAdd_cam,'IMAGE_EXT_ZMQ']);
                BGbeamblocked = BGbeamblocked+double(pull.data.val_val);
            end      
            waitfor(msgbox('remove beam block'));
        end
        
        

        [~] = doocswrite(config.doocsAdd_trigDelay,config.BGdelay);
        pause(2) 


        BGdelayoff=0;
        for i_avr=1:config.NumIMGs4average
            pull = doocsread([config.doocsAdd_cam,'IMAGE_EXT_ZMQ']);
            BGdelayoff = BGdelayoff+double(pull.data.val_val);
        end
        
        f_mainUI.UserData.profiles.(['zpos_',num2str(z_pos)]).BGbeamblocked = BGbeamblocked; 
        f_mainUI.UserData.profiles.(['zpos_',num2str(z_pos)]).BGdelayoff = BGdelayoff;     

        f = waitbar(0,'Please wait...');
        config.delays = linspace(config.centerBurstDelay-config.DelaySpan/2, config.centerBurstDelay+config.DelaySpan/2,config.N_pointsInBurst); 

        for i_delay = 1: length(config.delays)

            [~] = doocswrite(config.doocsAdd_trigDelay,config.delays(i_delay));
            pause(2.0) 

            
            img=0;
            for i_avr=1:config.NumIMGs4average
                pull = doocsread([config.doocsAdd_cam,'IMAGE_EXT_ZMQ']);
                img = img+double(pull.data.val_val);
            end

            f_mainUI.UserData.profiles.(['zpos_',num2str(z_pos)]).IMGraw.(['trigDelay_',num2str(config.delays(i_delay))])(:,:) = img;

            waitbar(i_delay/length(config.delays),f,['read cam img for delay ',num2str(config.delays(i_delay))]); 
            beamData(f_mainUI); 
        end
        close(f)

       [~] = doocswrite(config.doocsAdd_trigDelay,config.centerBurstDelay);
       z_pos = inputDLG('z_position',{'enter z position (integer in mm)'}) 
       
    end
end

function extractBeamProp(f_mainUI)
    profiles=f_mainUI.UserData.profiles;
    config=f_mainUI.UserData.config;
    
    positions = fieldnames(profiles);
    for i_pos = 1:length(positions)    
        posstr =positions{i_pos}(6:end);
        posstr(strfind(posstr,'p'))='.';
        results.beamProp.pos(i_pos) = str2double(posstr);
        results.delays = fieldnames(profiles.(positions{i_pos}).IMGraw);   
        for i_delay = 1 : length(results.delays)

            results.beamProp.rx_ISO(i_delay,i_pos)=config.pixelSize*profiles.(positions{i_pos}).beamParameters_ISO.(results.delays{i_delay}).rx;
            results.beamProp.ry_ISO(i_delay,i_pos)=config.pixelSize*profiles.(positions{i_pos}).beamParameters_ISO.(results.delays{i_delay}).ry;
            results.beamProp.Ip(i_delay,i_pos)=profiles.(positions{i_pos}).beamParameters_ISO.(results.delays{i_delay}).Ip;

            results.beamProp.rx_gauss(i_delay,i_pos)=config.pixelSize*mean(profiles.(positions{i_pos}).beamParameters_gaussfit.(results.delays{i_delay}).rx);
            results.beamProp.ry_gauss(i_delay,i_pos)=config.pixelSize*mean(profiles.(positions{i_pos}).beamParameters_gaussfit.(results.delays{i_delay}).ry);

            results.beamProp.crx_gauss(i_delay,i_pos)=config.pixelSize*diff(profiles.(positions{i_pos}).beamParameters_gaussfit.(results.delays{i_delay}).rx);
            results.beamProp.cry_gauss(i_delay,i_pos)=config.pixelSize*diff(profiles.(positions{i_pos}).beamParameters_gaussfit.(results.delays{i_delay}).ry);

        end
    end
    f_mainUI.UserData.results=results;
    
end

function results = UI_linfit(f_mainUI)
    results=f_mainUI.UserData.results;
    config=f_mainUI.UserData.config;

    T_ISO=table;
    T_GaussFit=table;
    for i_delay = 1 : length(results.delays)
        delay =  sscanf(results.delays{i_delay},'trigDelay_%f');

        h=figure(i_delay); 
        delete(h.Children(:));
        set(h,'Position',[h.Position(1:2)-i_delay*25 1000 h.Position(4)]);
        set(h,'Name',results.delays{i_delay});

        ax=subplot(2,1,1);
        [fx fy T]=LinearFit('ISO',delay,results.beamProp.pos',results.beamProp.rx_ISO(i_delay,:)',results.beamProp.ry_ISO(i_delay,:)',results.beamProp.Ip(i_delay,:),config,[],[]);
        T_ISO=[T_ISO; T];
        set(ax.Children,'ButtonDownFcn',{@ImageClickCallback,f_mainUI});
        
        ax=subplot(2,1,2);
        [fx fy T]=LinearFit('Guass fit',delay,results.beamProp.pos',results.beamProp.rx_gauss(i_delay,:)',results.beamProp.ry_gauss(i_delay,:)',results.beamProp.Ip(i_delay,:),config,results.beamProp.crx_gauss(i_delay,:)',results.beamProp.cry_gauss(i_delay,:)');
        T_GaussFit=[T_GaussFit; T];
        set(ax.Children,'ButtonDownFcn',{@ImageClickCallback,f_mainUI});

        if ~exist('LinearFitdata', 'dir'),  mkdir('LinearFitdata'); pause(0.5);  end
        saveas(h,fullfile(pwd,'LinearFitdata',['LinearFitdata',num2str(delay),'_BGsub',config.BGsub,'.jpg']),'jpg')

    end
    filename = fullfile(pwd,'M2data','results.xlsx');
    writetable(T_ISO,filename,'Sheet','ISO');
    writetable(T_GaussFit,filename,'Sheet','GaussFit');
    
    results.M2data = [T_ISO;T_GaussFit];

end


function results = getM2Data(f_mainUI)
    results=f_mainUI.UserData.results;
    config=f_mainUI.UserData.config;

    T_ISO=table;
    T_GaussFit=table;
    for i_delay = 1 : length(results.delays)
        delay =  sscanf(results.delays{i_delay},'trigDelay_%f');

        h=figure(i_delay); 
        delete(h.Children(:));
        set(h,'Position',[h.Position(1:2)-i_delay*25 1000 h.Position(4)]);
        set(h,'Name',results.delays{i_delay});

        ax=subplot(2,1,1);
        [fx fy T]=M2Fit('ISO',delay,results.beamProp.pos',results.beamProp.rx_ISO(i_delay,:)',results.beamProp.ry_ISO(i_delay,:)',results.beamProp.Ip(i_delay,:),config,[],[]);
        T_ISO=[T_ISO; T];
        set(ax.Children,'ButtonDownFcn',{@ImageClickCallback,f_mainUI});
        
        ax=subplot(2,1,2);
        [fx fy T]=M2Fit('Guass fit',delay,results.beamProp.pos',results.beamProp.rx_gauss(i_delay,:)',results.beamProp.ry_gauss(i_delay,:)',results.beamProp.Ip(i_delay,:),config,results.beamProp.crx_gauss(i_delay,:)',results.beamProp.cry_gauss(i_delay,:)');
        T_GaussFit=[T_GaussFit; T];
        set(ax.Children,'ButtonDownFcn',{@ImageClickCallback,f_mainUI});

        if ~exist('M2data', 'dir'),  mkdir('M2data'); pause(0.5);  end
        saveas(h,fullfile(pwd,'M2data',['M2data_',num2str(delay),'_BGsub',config.BGsub,'.jpg']),'jpg')

    end
    filename = fullfile(pwd,'M2data','results.xlsx');
    writetable(T_ISO,filename,'Sheet','ISO');
    writetable(T_GaussFit,filename,'Sheet','GaussFit');
    
    results.M2data = [T_ISO;T_GaussFit];

end


function ImageClickCallback(ax, event, f_mainUI)

    cpt = get(gca,'CurrentPoint');
    pt = cpt(1,1:2);
   
    
    %%find pos data point
    pos=f_mainUI.UserData.results.beamProp.pos;
    pos=pos(abs(pos-pt(1)*1e3)==min(abs(pos-pt(1)*1e3)))
    pos=pos(1);
    
    
    posstr=num2str(pos);
    posstr(strfind(posstr,'.'))='p';
    f_mainUI.UserData.profiles.(['zpos_',posstr]).checkSums.(ax.Parent.Parent.Name)=rand();
    beamData(f_mainUI);
    h=gcf;%figure;
%     temp=f_mainUI.UserData.profiles.(['zpos_',num2str(pos)]);
%     imshow(temp.previewPlot.( ax.Parent.Parent.Name).imind,temp.previewPlot.( ax.Parent.Parent.Name).cm)
    
    btn_h = 20;
    i_btn=1;    
    uicontrol(h,'Style','pushbutton','String','delete data point','Position',[250 1 150 20],...
        'Callback', {@UI_delDataPoint,f_mainUI,['zpos_',posstr]});

    i_btn=i_btn+1;    
    uicontrol(h,'Style','pushbutton','String','save raw','Position',[400 1 150 20],...
        'Callback', {@UI_saveRaw,f_mainUI.UserData.profiles.(['zpos_',posstr]).IMGraw.(ax.Parent.Parent.Name)});

end

function UI_delDataPoint(ax,event,f_mainUI,zpos)

    f_mainUI.UserData.profiles = rmfield(f_mainUI.UserData.profiles,zpos);
    
    extractBeamProp(f_mainUI);
    getM2Data(f_mainUI);

end



function UI_saveRaw(ax,event,rawIMG)

    [baseName, folder] = uiputfile('.dat','save raw image');   if baseName==0; return , end
    
    fullpath = fullfile(folder, baseName);
    
   
    waitMsg(1);
    rawIMG = double(rawIMG);
    save(fullpath,'rawIMG','-ASCII');   
    waitMsg(0);
    
end

function beamData(f_mainUI)

    persistent ax_text
    profiles=f_mainUI.UserData.profiles;
    config  =f_mainUI.UserData.config;

    if isempty(profiles)
        return
    end

    positions = fieldnames(profiles);
% tic    

    for i_pos = 1:length(positions)    

        delays = fieldnames(profiles.(positions{i_pos}).IMGraw);   
        for i_delay = 1 : length(delays)
            
            profile = profiles.(positions{i_pos}).IMGraw.(delays{i_delay});
            
            switch config.BGsub
                case 'delay'
                    if isfield(profiles.(positions{i_pos}),'BGdelayoff')
                        profile = profile-profiles.(positions{i_pos}).BGdelayoff;
                    end
                case 'blocked'
                    if isfield(profiles.(positions{i_pos}),'BGbeamblocked')
                        profile = profile-profiles.(positions{i_pos}).BGbeamblocked;
                    end
            end
            
            checkSum = sum(sum(profile(1:3:end,1:3:end)))...
                +config.aperture2diameter...
                +config.principalAxesRot...
                +config.pixelSize;
            
%          
            if ~isfield(profiles.(positions{i_pos}),'checkSums') || ...     
               ~isfield(profiles.(positions{i_pos}).checkSums,delays{i_delay}) || ...          
                checkSum  ~= profiles.(positions{i_pos}).checkSums.(delays{i_delay})
            
                f_mainUI.UserData.profiles.(positions{i_pos}).checkSums.(delays{i_delay}) = checkSum;  
                
     
                profile = BGsub(double(profile)); %simple background substruction
                
           
    %             [rx ry cx cy] = getBeamSize(profile);

                %[cy cx]=find(profile==max(profile(:)),1);
                profile_s = imgaussfilt(double(profile), 2);    % smooth to kill hot pixels
                [cy, cx]  = find(profile_s == max(profile_s(:)), 1);

                Ix = profile(cy,:); Ix=Ix/max(Ix(:));
                Iy = profile(:,cx); Iy=Iy/max(Iy(:));

                rx=max(find(Ix>0.5)) - min(find(Ix>0.5));
                ry=max(find(Iy>0.5)) - min(find(Iy>0.5));

                rx_ = rx;
                ry_ = ry;

                Rx=[];
                Ry=[];

                [Ny Nx] = size(profile);
    %             tic

                for i_ = 1:100   


                    
                    xmin = round(cx-rx*config.aperture2diameter); if(xmin<1) ,xmin=1  ;end
                    xmax = round(cx+rx*config.aperture2diameter); if(xmax>Nx),xmax=Nx ;end

                    ymin = round(cy-ry*config.aperture2diameter); if(ymin<1) ,ymin=1  ;end
                    ymax = round(cy+ry*config.aperture2diameter); if(ymax>Ny),ymax=Ny ;end                
                    %%
                    processedIMG = BGsub(profile(ymin:ymax,xmin:xmax));
                    
                    [rx ry cx cy phi] = getBeamSize(processedIMG);

                    cx=cx+xmin;
                    cy=cy+ymin;
                
%                     disp(abs(rx - rx_) + abs(ry - ry_))
                    if abs(rx - rx_) + abs(ry - ry_) < 1e-1
                        break
                    else
                        rx_ = rx;
                        ry_ = ry;
                    end                

%                     Rx(end+1)=rx;
%                     Ry(end+1)=ry;
%                     figure(1111)
%                     plot(1:i_,Rx,...
%                          1:i_,Ry),drawnow


                end

                if i_==20
                    disp(['sigma and aperture for ', positions{i_pos},' / delay ',num2str(i_delay),' is not converging'])
                end
                %%
                [Ny_ Nx_] = size(processedIMG);
                [X,Y] = meshgrid(1:Nx_,1:Ny_);
                processedIMG = processedIMG/sum(processedIMG(:));
                
                %%binning for peak intensity measurment
                %binSize is chosen to keep number of pixels with in the
                %aperture smaller than 20 pixels within the beam diameter
                Nbin=20*config.aperture2diameter;
                
          
                %make processedIMG squared
                Ntmp=length(processedIMG);
                binSize = ceil(Ntmp/Nbin);                
                Ntmp=binSize*Nbin;
                
                sqIMG=zeros(Ntmp);    
                sqIMG(1:Ny_,1:Nx_)=processedIMG;
                
                
                if binSize > 1
                    tmp = sum(reshape(sqIMG,binSize,[]));
                    tmp = reshape(tmp,size(sqIMG,1) / binSize,[])';
                    tmp = sum(reshape(tmp,binSize,[]));
                    tmp = reshape(tmp,size(sqIMG,2) / binSize,[])';
                    tmp = tmp/sum(tmp(:));
				else
					tmp = sqIMG;
                end
                tmp = tmp/sum(tmp(:));
                Ip = max(tmp(:))/binSize^2;

                %ISO beam radia in beams principal planes
                if config.principalAxesRot
                    rotIMG = rotateIMG(processedIMG,phi,profile,xmin,xmax,ymin,ymax);
                else
                    rotIMG = processedIMG;
                    phi = 0;
                end
                [rx_rot ry_rot cx_rot cy_rot phi_rot] = getBeamSize(rotIMG);
                        xmin_rot = xmin-(size(rotIMG,2)-Nx_)/2;
                        ymin_rot = ymin-(size(rotIMG,1)-Ny_)/2;
                        xmax_rot = xmax+(size(rotIMG,2)-Nx_)/2;
                        ymax_rot = ymax+(size(rotIMG,1)-Ny_)/2; 
                        cx_rot = cx_rot + xmin_rot;
                        cy_rot = cy_rot + ymin_rot;                        
                
                %%gaussfits in beams principal planes
                Ix = sum(rotIMG,1);
                Iy = sum(rotIMG,2)';

                [fx AmpCenRadx] = fitGauss([xmin_rot:xmax_rot]+1,Ix,[max(Ix) cx_rot rx_rot]);
                [fy AmpCenRady] = fitGauss([ymin_rot:ymax_rot]+1,Iy,[max(Iy) cy_rot ry_rot]);




                %%
                h=figure(100);
                h.Name='Beam analysis';
                delete(h.Children(:))
                tmpstr=positions{i_pos};
                tmpstr(1:5)=[];
                tmpstr(strfind(tmpstr,'p'))='.';
                h.Name=[tmpstr,' | ', delays{i_delay}];
                
                %%
                subplot(2,2,3)
                               
                x = [1:Nx_]-(cx-xmin);
                y = [1:Ny_]-(cy-ymin);
                image(x,y,processedIMG,'CDataMapping','scaled')
                
                if exist('ax1') && ishandle(ax1), delete(ax1),end
                ax1=ellipse(fx.radius,fy.radius,phi,0,0,'g');                
                %
                
                if exist('ax2') && ishandle(ax2), delete(ax2),end
                ax2=ellipse(rx_rot,ry_rot,phi,0,0,'r');
                
                if exist('axpx') && ishandle(axpx), delete(axpx),end
                axpx=ellipse(fx.radius*10,0,phi,0,0,[0 0 0],10);
                
                if exist('axpy') && ishandle(axpy), delete(axpy),end
                axpy=ellipse(0,fy.radius*10,phi,0,0,[0 0 0],10);
                %
                colormap(vertcat([1,1,1],[.9,.9,.9],[.8,.8,.8],jet))
                xlim([min([x y]) max([x y])])
                ylim([min([x y]) max([x y])])
                xlabel('x (pixels)')
                ylabel('y (pixels)')
                title(sprintf('beam profile\n labratory frame'))
                axis('square')
                %
                if exist('axx') && ishandle(axx), delete(axx),end
                axx=subplot(2,2,1);
                x=[xmin_rot:xmax_rot]-cx;
                plot((x),fx(x+cx)/max(Ix(:)),'b','LineWidth',3);,hold on
                plot((x),Ix/max(Ix(:)),'.r'),hold off
                xlim([min([x y]) max([x y])])
                ylim([0 1])
                axx.YTick=[]  ;
                title(sprintf('Integrated profile\n principal x-axis'))
                axis('square')
                %
                if exist('axy') && ishandle(axy), delete(axy),end
                axy=subplot(2,2,4);
                y=[ymin_rot:ymax_rot]-cy;
                plot((y),fy(y+cy)/max(Iy(:)),'b','LineWidth',3);,hold on
                plot((y),Iy/max(Iy(:)),'.r'),hold off
                xlim([min([x y]) max([x y])])
                ylim([0 1])
                axy.YTick=[]  ;              
                title(sprintf('Integrated profile\n beams principal y-axis'))
                axis('square')
                
                camroll(-90)
                
                %
                if exist('ax_text') && isempty(ax_text) || ishandle(ax_text), delete(ax_text),end
                subplot(2,2,2);
                
                ax_text=text(0,0.5,'');              
           
                set(ax_text,'String',...
                sprintf(['z position = ',tmpstr,' mm \n',...
                         'trig delay = ',delays{i_delay}(11:end),' us\n\n',...
                         'principal axis = ',num2str(phi,2),' rad\n',...
                    'Gauss-fit (blue):\n',...
                    'radius_x = ',num2str(mean(AmpCenRadx(:,3))*config.pixelSize,'%.2e'),char(177),num2str(diff(AmpCenRadx(:,3)*config.pixelSize)/2,'%.2e'),' m\n',...
                    'radius_y = ',num2str(mean(AmpCenRady(:,3))*config.pixelSize,'%.2e'),char(177),num2str(diff(AmpCenRady(:,3)*config.pixelSize)/2,'%.2e'),' m\n',...
                    'ISO (red):\n',...
                    'radius_x = ',num2str(rx*config.pixelSize,'%.2e'),' m\n',...
                    'radius_y = ',num2str(ry*config.pixelSize,'%.2e'),' m\n',...        
                    ]),'FontSize',10);
            %     title(fileName,'Interpreter', 'none','FontSize',8)
                axis off                
                
                %%
                if ~exist('processedIMGs', 'dir'),  mkdir('processedIMGs'); pause(0.5);  end
                saveas(h,fullfile(pwd,'processedIMGs',['processedImg_',positions{i_pos},'_',delays{i_delay},'.jpg']),'jpg')
                [previewPlot.imind,previewPlot.cm] = rgb2ind(frame2im(getframe(h)),256);
                %%
                f_mainUI.UserData.profiles.(positions{i_pos}).previewPlot.(delays{i_delay}) = previewPlot;

                f_mainUI.UserData.profiles.(positions{i_pos}).beamParameters_ISO.(delays{i_delay}).cx = cx;
                f_mainUI.UserData.profiles.(positions{i_pos}).beamParameters_ISO.(delays{i_delay}).cy = cy;
                f_mainUI.UserData.profiles.(positions{i_pos}).beamParameters_ISO.(delays{i_delay}).rx = rx;
                f_mainUI.UserData.profiles.(positions{i_pos}).beamParameters_ISO.(delays{i_delay}).ry = ry;
                f_mainUI.UserData.profiles.(positions{i_pos}).beamParameters_ISO.(delays{i_delay}).Ip = Ip;   
                
                
                f_mainUI.UserData.profiles.(positions{i_pos}).beamParameters_gaussfit.(delays{i_delay}).cx = AmpCenRadx(:,2);
                f_mainUI.UserData.profiles.(positions{i_pos}).beamParameters_gaussfit.(delays{i_delay}).cy = AmpCenRady(:,2);
                f_mainUI.UserData.profiles.(positions{i_pos}).beamParameters_gaussfit.(delays{i_delay}).rx = AmpCenRadx(:,3);
                f_mainUI.UserData.profiles.(positions{i_pos}).beamParameters_gaussfit.(delays{i_delay}).ry = AmpCenRady(:,3);   
           
            end
            
        end
    end
    
% toc


end

function IMG = rotateIMG(IMG,phi,fullIMG,xmin,xmax,ymin,ymax)

    [Ny_full, Nx_full] = size(fullIMG);
    [Ny_roi, Nx_roi] = size(IMG);

    rotDeg = phi * 180/pi;
    abs_cos = abs(cos(phi));
    abs_sin = abs(sin(phi));

    rot_h = max(Ny_roi, ceil(Ny_roi * abs_cos + Nx_roi * abs_sin) + 2);
    rot_w = max(Nx_roi, ceil(Ny_roi * abs_sin + Nx_roi * abs_cos) + 2);

    y_offset = floor((rot_h - Ny_roi)/2);
    x_offset = floor((rot_w - Nx_roi)/2);

    ymin_exp = ymin - y_offset;
    xmin_exp = xmin - x_offset;
    ymax_exp = ymin_exp + rot_h - 1;
    xmax_exp = xmin_exp + rot_w - 1;

    expanded_patch = zeros(rot_h, rot_w);

    y_in_start = max(ymin_exp, 1);
    y_in_end = min(ymax_exp, Ny_full);
    x_in_start = max(xmin_exp, 1);
    x_in_end = min(xmax_exp, Nx_full);

    if y_in_start <= y_in_end && x_in_start <= x_in_end
        patch_y_start = y_in_start - ymin_exp + 1;
        patch_y_end = patch_y_start + (y_in_end - y_in_start);
        patch_x_start = x_in_start - xmin_exp + 1;
        patch_x_end = patch_x_start + (x_in_end - x_in_start);

        expanded_patch(patch_y_start:patch_y_end, patch_x_start:patch_x_end) = ...
            fullIMG(y_in_start:y_in_end, x_in_start:x_in_end);
    end

    expanded_patch(y_offset+1:y_offset+Ny_roi, x_offset+1:x_offset+Nx_roi) = IMG;

    target_sum = sum(IMG(:));

    rotated_full = imrotate(expanded_patch, rotDeg, 'bilinear', 'crop');

    start_y = y_offset + 1;
    end_y = start_y + Ny_roi - 1;
    start_x = x_offset + 1;
    end_x = start_x + Nx_roi - 1;

    IMG = rotated_full(start_y:end_y, start_x:end_x);

    orig_sum = sum(IMG(:));
    if target_sum ~= 0 && orig_sum ~= 0
        IMG = IMG * (target_sum / orig_sum);
    end

end

function [f AmpCenRad] = fitGauss(X,I,startparam)

    model='A*exp(-2*((x-center)/radius)^2)';
    f = fit(X',I',model,fitoptions(model,'StartPoint',startparam)); %figure(22),plot(f),hold on,plot(I),hold off
    AmpCenRad =confint(f);

end

function [rx ry cx cy phi]  = getBeamSize(profile)
    persistent X Y Nx_ Ny_

    [Ny Nx] = size(profile);
    
    if isempty(Nx_) || isempty(Ny_) || Nx ~= Nx_ || Ny ~= Ny_
        [X,Y] = meshgrid(1:Nx,1:Ny);
    end
    
    Nx_ =Nx;
    Ny_ =Ny;
    
    Int = sum(profile(:));
    
    cx = sum(profile(:).*X(:))/Int;
    cy = sum(profile(:).*Y(:))/Int;

    sigma_x_sq  = sum(  profile(:) .* (X(:)-cx).^2  )/Int  ;
    sigma_y_sq  = sum(  profile(:) .* (Y(:)-cy).^2  )/Int  ;
    sigma_xy_sq = sum(  profile(:) .* (X(:)-cx).*(Y(:)-cy)  )/Int  ;
    
    rx = 2*sqrt( abs( sigma_x_sq));
    ry = 2*sqrt( abs( sigma_y_sq));    
%     phi = 0.5 * atan2(2* sign(sigma_xy_sq)* (abs(sigma_xy_sq)) , (sigma_x_sq - sigma_y_sq)   )+pi/2;   
    if sigma_x_sq ~= sigma_y_sq
        phi = 0.5 * atan(2* sign(sigma_xy_sq)* (abs(sigma_xy_sq)) / (sigma_x_sq - sigma_y_sq)   );   
    else
        phi = pi/4*sigma_xy_sq/abs(sigma_xy_sq);
    end
end

function profile=BGsub(profile)

%         N=round(min(size(profile)) *0.1);
%       
%         BG1=mean(sum(profile([1:N end-N:end],:),2))/size(profile,2);
%         BG2=mean(sum(profile(:,[1:N end-N:end]),1))/size(profile,1);
%         profile=profile-mean([BG1 BG2]);

    %N=ceil(min(size(profile)) *0.01);
    %BG = profile([1:N,end-N:end   ],[1:N,end-N:end  ]);
    %profile=profile-mean(BG(:));
    %figure(2),plot(profile)

    P = double(profile);
    m = ceil(0.05*min(size(P)));                       % 5% border
    B = P([1:m,end-m+1:end],[1:m,end-m+1:end]);       % four corners
    b = B(:);
    for i=1:3                                          % 3x sigma-clipping
        mu = median(b); sig = 1.4826*mad(b,1);
        b = b(abs(b-mu) < 3*max(sig,eps));
    end
    bg = median(b);                                    % robust background level
    profile = P - bg;
end

function output = inputDLG(dlgtitle,prompt)
    
    persistent definput
    
    if ~isfield(definput,dlgtitle) || isempty(definput.(dlgtitle))
        answer = inputdlg(prompt,dlgtitle,[1 35]);
    else
        answer = inputdlg(prompt,dlgtitle,[1 35],definput.(dlgtitle));
    end
    
    definput.(dlgtitle) = answer;
    
    
    output = str2double(answer);

    if any(isnan(output))   
         output = inputDLG(dlgtitle,prompt);
    end
     
    output = round(output);

end

function h = plotEllipses(cnt,rads,axh)
    % cnt is the [x,y] coordinate of the center (row or column index).
    % rads is the [horizontal, vertical] "radius" of the ellipses (row or column index).
    % axh is the axis handle (if missing or empty, gca will be used)
    % h is the object handle to the plotted rectangle.
    % The easiest approach IMO is to plot a rectangle with rounded edges. 
    % EXAMPLE
    %    center = [1, 2];         %[x,y] center (mean)
    %    stdev = [1.2, 0.5];      %[x,y] standard dev.
    %    h = plotEllipses(center, stdev)
    %    axis equal
    % get axis handle if needed
    if nargin < 3 || isempty(axh)
       axh = gca();  
    end
    % Compute the lower, left corner of the rectangle.
    llc = cnt(:)-rads(:);
    % Compute the width and height
    wh = rads(:)*2; 
    % Draw rectangle 
    h = rectangle(axh,'Position',[llc(:).',wh(:).'],'Curvature',[1,1]); 
end

function [fx fy T]=M2Fit(Type,trigDelay,position,rx,ry,Ip,config,crx_gauss,cry_gauss)

        position=position*1e-3;
        lambda = config.wavelength;
        pixelsize = config.pixelSize;
        %%
        model=['w0.*sqrt( 1 + (  M_sq .* ',num2str(lambda),' .* (x - z0)/(pi.*w0.^2)).^2   )'];
        
        fx = fit(position,rx,model,fitoptions(model,'StartPoint',[1 min(rx) mean(position(rx==min(rx)))],'Robust','on')); 
        fy = fit(position,ry,model,fitoptions(model,'StartPoint',[1 min(ry) mean(position(ry==min(ry)))],'Robust','on')); 
        
        try
            px =confint(fx);
            py =confint(fy);
        catch
            px = [0 0 0; 1 1 1];
            py = [0 0 0; 1 1 1];
            msgbox('Cannot compute confidence intervals if #observations<=#coefficients.')
        end
            
        if isempty(crx_gauss)
            plot(position,rx,'or'),hold on,plot(fx,'-r')
            plot(position,ry,'ob'),hold on,plot(fy,'-b'),hold off
        else
            errorbar(position,rx,crx_gauss,'or'),hold on,plot(fx,'-r')
            errorbar(position,ry,cry_gauss,'ob'),hold on,plot(fy,'-b'),hold off
        end

        title(Type)
        xlabel('position (m)')
        ylabel('radius 1/e2 (m)')
        
        M2_x=fx.M_sq;
        M2_y=fy.M_sq;
        z0_x=fx.z0;
        z0_y=fy.z0;
        w0_x=fx.w0;
        w0_y=fy.w0;
        dM2_x=diff(px(:,1)/2);
        dM2_y=diff(py(:,1)/2);
        dz0_x=diff(px(:,3)/2);
        dz0_y=diff(py(:,3)/2);
        dw0_x=diff(py(:,2)/2);
        dw0_y=diff(py(:,2)/2);        
        zr_x    =pi/lambda *fx.w0^2 /fx.M_sq;
        zr_y    =pi/lambda *fy.w0^2 /fy.M_sq;
        theta_x =fx.w0 /zr_x;
        theta_y =fy.w0 /zr_y;
        
        w_z_x = w0_x*sqrt(1 + ((position-z0_x)/zr_x).^2)';
        w_z_y = w0_y*sqrt(1 + ((position-z0_y)/zr_y).^2)';
        
        Ip_gauss = 2./(pi*w_z_x.*w_z_y/(pixelsize^2));
        Ip_enhanced = Ip./Ip_gauss;
        
        yyaxis right
        plot(position,Ip_enhanced,'p');%,'MarkerEdgeColor', 	[1 0 1]);
        ylabel('peak intensity enhancment')
        
        legend('x-data:',sprintf(['M2 =',num2str(fx.M_sq,'%.2f'),char(177),num2str(diff(px(:,1)/2),'%.2f'),'\nz0 = ',num2str(fx.z0,'%.2e'),char(177),num2str(diff(px(:,3)/2),'%.0e m'),'\nw0 = ',num2str(fx.w0,'%.2e '),char(177),num2str(diff(px(:,2)/2),'%.0e m'),'\nzr = ',num2str(zr_x,'%.2e m'),'\ntheta = ',num2str(theta_x,'%.2e')]),...
               'y-data:',sprintf(['M2 =',num2str(fy.M_sq,'%.2f'),char(177),num2str(diff(py(:,1)/2),'%.2f'),'\nz0 = ',num2str(fy.z0,'%.2e'),char(177),num2str(diff(py(:,3)/2),'%.0e m'),'\nw0 = ',num2str(fy.w0,'%.2e '),char(177),num2str(diff(py(:,2)/2),'%.0e m'),'\nzr = ',num2str(zr_y,'%.2e m'),'\ntheta = ',num2str(theta_y,'%.2e')]),...
               sprintf('peak intensity enhancment\n(click for beam profile)'),...
               'Location','bestoutside','FontSize',12) %bestoutside
        legend boxoff  
        T = table({Type},trigDelay,M2_x,dM2_x,M2_y,dM2_y,z0_x,dz0_x,z0_y,dz0_y,w0_x,dw0_x,w0_y,dw0_y,zr_x,zr_y,theta_x,theta_y);
end


function [fx fy T]=LinearFit(Type,trigDelay,position,rx,ry,Ip,config,crx_gauss,cry_gauss)

        position=position*1e-3;
        pixelsize = config.pixelSize;
        %%
%         model=['w0.*sqrt( 1 + (  M_sq .* ',num2str(lambda),' .* (x - z0)/(pi.*w0.^2)).^2   )'];
        model=['m*x + n'];
        fx = fit(position,rx,model,fitoptions(model,'StartPoint',[0 min(rx)],'Robust','on')); 
        fy = fit(position,ry,model,fitoptions(model,'StartPoint',[0 min(ry)],'Robust','on')); 
        
        try
            px =confint(fx);
            py =confint(fy);
        catch
            px = [0 0 0; 1 1 1];
            py = [0 0 0; 1 1 1];
            msgbox('Cannot compute confidence intervals if #observations<=#coefficients.')
        end
            
        if isempty(crx_gauss)
            plot(position,rx,'or'),hold on,plot(fx,'-r')
            plot(position,ry,'ob'),hold on,plot(fy,'-b'),hold off
        else
            errorbar(position,rx,crx_gauss,'or'),hold on,plot(fx,'-r')
            errorbar(position,ry,cry_gauss,'ob'),hold on,plot(fy,'-b'),hold off
        end

        title(Type)
        xlabel('position (m)')
        ylabel('radius 1/e2 (m)')
        

        z0waist_x=fx.n;
        z0waist_y=fy.n;
        dz0waist_x=diff(px(:,2)/2);
        dz0waist_y=diff(py(:,2)/2);          
        
        theta_x =fx.m;
        theta_y =fy.m;
        dtheta_x=diff(px(:,1)/2);
        dtheta_y=diff(py(:,1)/2);
        
        w_z_x = theta_x*position' + z0waist_x;
        w_z_y = theta_y*position' + z0waist_y;
      
        
        Ip_gauss = 2./(pi*w_z_x.*w_z_y/(pixelsize^2));
        Ip_enhanced = Ip./Ip_gauss;
        
        yyaxis right
        plot(position,Ip_enhanced,'p');%,'MarkerEdgeColor', 	[1 0 1]);
        ylabel('peak intensity enhancment')
        
        legend('x-data:',sprintf(['theta =',num2str(theta_x,'%.2e'),char(177),num2str(diff(px(:,1)/2),'%.2e'),' rad\nz0waist = ',num2str(z0waist_x,'%.2e'),char(177),num2str(diff(px(:,2)/2),'%.0e m')]),...
               'y-data:',sprintf(['theta =',num2str(theta_y,'%.2e'),char(177),num2str(diff(py(:,1)/2),'%.2e'),' rad\nz0waist = ',num2str(z0waist_y,'%.2e'),char(177),num2str(diff(py(:,2)/2),'%.0e m')]),...
               sprintf('peak intensity enhancment\n(click for beam profile)'),...
               'Location','bestoutside','FontSize',12) %bestoutside
        legend boxoff  
        T = table({Type},trigDelay,theta_x,theta_y,dtheta_x,dtheta_y,z0waist_x,z0waist_x,dz0waist_x,dz0waist_x);
end

function h=ellipse(ra,rb,ang,x0,y0,C,Nb)
    % Ellipse adds ellipses to the current plot
    %
    % ELLIPSE(ra,rb,ang,x0,y0) adds an ellipse with semimajor axis of ra,
    % a semiminor axis of radius rb, and an orientation of the semimajor
    % axis with an angle of ang (in radians) rotated counter-clockwise 
    % from the x-axis.  The ellipse is centered at the point x0,y0.
    %
    % The length of ra, rb, and ang should be the same. 
    % If ra is a vector of length L and x0,y0 scalars, L ellipses
    % are added at point x0,y0.
    % If ra is a scalar and x0,y0 vectors of length M, M ellipse are with the same 
    % radii are added at the points x0,y0.
    % If ra, x0, y0 are vectors of the same length L=M, M ellipses are added.
    % If ra is a vector of length L and x0, y0 are  vectors of length
    % M~=L, L*M ellipses are added, at each point x0,y0, L ellipses of radius ra.
    %
    % ELLIPSE(ra,rb,ang,x0,y0,C)
    % adds ellipses of color C. C may be a string ('r','b',...) or the RGB value. 
    % If no color is specified, it makes automatic use of the colors specified by 
    % the axes ColorOrder property. For several ellipses C may be a vector.
    %
    % ELLIPSE(ra,rb,ang,x0,y0,C,Nb), Nb specifies the number of points
    % used to draw the ellipse. The default value is 300. Nb may be specified
    % for each ellipse individually, in which case it should be the same
    % length as ra, rb, etc.
    %
    % h=ELLIPSE(...) returns the handles to the ellipses.
    %
    % usage exmple: the following produces a red ellipse centered at 1,1
    % and tipped down at a 45 deg axis from the x axis
    % ellipse(1,2,pi/4,1,1,'r')
    %
    % note that if ra=rb, ELLIPSE plots a circle
    %
    % written by D.G. Long, Brigham Young University, based on the
    % CIRCLES.m original written by Peter Blattner, Institute of 
    % Microtechnology, University of Neuchatel, Switzerland, blattner@imt.unine.ch
    % Check the number of input arguments 
    if nargin<1,
      ra=[];
    end;
    if nargin<2,
      rb=[];
    end;
    if nargin<3,
      ang=[];
    end;
    if nargin<5,
      x0=[];
      y0=[];
    end;

    if nargin<6,
      C=[];
    end
    if nargin<7,
      Nb=[];
    end
    % set up the default values
    if isempty(ra),ra=1;end;
    if isempty(rb),rb=1;end;
    if isempty(ang),ang=0;end;
    if isempty(x0),x0=0;end;
    if isempty(y0),y0=0;end;
    if isempty(Nb),Nb=300;end;
    if isempty(C),C=get(gca,'colororder');end;
    % work on the variable sizes
    x0=x0(:);
    y0=y0(:);
    ra=ra(:);
    rb=rb(:);
    ang=ang(:);
    Nb=Nb(:);
    if isstr(C),C=C(:);end;
    if length(ra)~=length(rb),
      error('length(ra)~=length(rb)');
    end;
    if length(x0)~=length(y0),
      error('length(x0)~=length(y0)');
    end;
    % how many inscribed elllipses are plotted
    if length(ra)~=length(x0)
      maxk=length(ra)*length(x0);
    else
      maxk=length(ra);
    end;
    % drawing loop
    for k=1:maxk

      if length(x0)==1
        xpos=x0;
        ypos=y0;
        radm=ra(k);
        radn=rb(k);
        if length(ang)==1
          an=ang;
        else
          an=ang(k);
        end;
      elseif length(ra)==1
        xpos=x0(k);
        ypos=y0(k);
        radm=ra;
        radn=rb;
        an=ang;
      elseif length(x0)==length(ra)
        xpos=x0(k);
        ypos=y0(k);
        radm=ra(k);
        radn=rb(k);
        an=ang(k)
      else
        rada=ra(fix((k-1)/size(x0,1))+1);
        radb=rb(fix((k-1)/size(x0,1))+1);
        an=ang(fix((k-1)/size(x0,1))+1);
        xpos=x0(rem(k-1,size(x0,1))+1);
        ypos=y0(rem(k-1,size(y0,1))+1);
      end;
      % draw ellipse

      co=cos(an);
      si=sin(an);
      the=linspace(0,2*pi,Nb(rem(k-1,size(Nb,1))+1,:)+1);
      %  x=radm*cos(the)*co-si*radn*sin(the)+xpos;
      %  y=radm*cos(the)*si+co*radn*sin(the)+ypos;
      p=line(radm*cos(the)*co-si*radn*sin(the)+xpos,radm*cos(the)*si+co*radn*sin(the)+ypos);
      set(p,'color',C(rem(k-1,size(C,1))+1,:));

      % output handles to each ellipse if output variable specified

      if nargout > 0
        h(k)=p;
      end

    end
end
