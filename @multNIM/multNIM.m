classdef multNIM
% Class implementation of multiplicative NIM based on NIM class. 

properties
	nim;		    % NIM or sNIM struct that contains the normal subunits
	Msubunits;  % actual NIM/sNIM subunits that should multiply Mtargets
	Mtargets;   % targets of Msubunits in the NIM
end	

properties (Hidden)
	version = '0.3';    %source code version used to generate the model
	create_on = date;   %date model was generated
end	

%% ******************** constructor ********************************
methods

	function mnim = multNIM( nim, Msubunits, Mtargets )
	% Usage: mnim = multNIM( nim, <Msubunits, Mtargets> )
	%
	% INPUTS:
	%   nim:        either an object of the NIM or sNIM class
	%   Msubunits:  array of SUBUNIT objects that will act as multiplicative
	%               gains on the subunits of the nim object
	%   Mtargets:   array of integers if each subunit only targets a single
	%               additive subunit, or a cell array, where each cell
	%               contains the targets of the corresponding Msubunit
	% 
	% OUTPUTS:
	%   mnim:       initialized multNIM object
	
		% Handle the no-input-argument case by returning a null model. This is important when initializing arrays of objects
		if nargin == 0
			return 
		end
		if nargin < 2
			return
		end

    % define defaults
    mnim.nim = nim;
    mnim.Msubunits = [];
    mnim.Mtargets = [];

    % turn Mtargets into a cell array if not already
    if ~iscell(Mtargets)
        Mtargets = num2cell(Mtargets);
    end

    % error checking on inputs
    assert( nargin == 3, 'Must specify targets as well as subunits' )
    assert( all(cellfun(@(x) ismember(x,1:length(nim.subunits)),Mtargets)), 'Invalid Mtargets.' )
    mnim.Msubunits = Msubunits;
    mnim.Mtargets = Mtargets;
    end

end
%% ********************  fitting methods ********************************
methods
	
	function mnim_out = fit_Afilters( mnim, Robs, stims, varargin )
	% Usage: mnim = mnim.fit_Afilters( Robs, stims, Uindx, varargin )
	%
    % Fits filters of additive subunits
    
    % ensure proper format of stims cell array
    if ~iscell(stims)
        tmp = stims;
        clear stims
        stims{1} = tmp;
    end

    % append necessary options to varargin to pass to fit_filters
    varargin{end+1} = 'gain_funs';
    varargin{end+1} = mnim.calc_gmults( stims );
    %varargin{end+1} = 'fit_offsets';
    %varargin{end+1} = 1;

    % fit filters
    mnim_out = mnim;
    mnim_out.nim = mnim.nim.fit_filters( Robs, stims, varargin{:} );
        
	end
	
	function mnim_out = fit_Mfilters( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_Mfilters( Robs, stims, Uindx, varargin )
	%
	% Fits filters of multiplicative subunit
	% Note 1: Specify which Msubunits to optimize using 'subs' option, 
    % numbered by their index in Msubunits array
    % Note 2: method only handles case where the Msubunits to optimize
    % target unique additive subunits; if this is not the case, use
    % fit_Msequential (which will in turn call this method properly)
    
    % ensure proper format of stims cell array
    if ~iscell(stims)
        tmp = stims;
        clear stims
        stims{1} = tmp;
    end

    % parse optional arguments, separating those needed in this method
    % from those needed as inputs to fit_filters method in NIM
    [~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs'} );
    % save indices of subunits targeted for fitting
    if isfield( parsed_inputs, 'subs' )
        Mtar = parsed_inputs.subs;
    else
        Mtar = 1:length( mnim.Msubunits );
    end
    assert(all(ismember(Mtar,1:length(mnim.Msubunits))),'Invalid ''subs'' input')
    NMsubs = length(Mtar);

    % ensure multiple multiplicative units are not acting on the same
    % additive subunit
    Atargs = [];
    for i = 1:NMsubs
        temp_targs = mnim.Mtargets{Mtar(i)};
        Atargs = [Atargs; temp_targs(:)];
    end
    assert(length(Atargs)==length(unique(Atargs)),'Cannot simultaneously fit two mult subunits with same target; use fit_Msequential method')

    % swap roles of additive and multiplicative subunits
    [nim_swap,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtar );

    % append necessary options to varargin to pass to fit_filters
    modvarargin{end+1} = 'gain_funs';
    modvarargin{end+1} = gmults;
    modvarargin{end+1} = 'subs';
    modvarargin{end+1} = 1:NMsubs;
    %modvarargin{pos+4} = 'fit_offsets';
    %modvarargin{pos+5} = 1;

    % fit filters using NIM method
    nim_swap = nim_swap.fit_filters( Robs, stims_plus, modvarargin{:} );

    % Copy filters back to their locations
    mnim_out = mnim;
    mnim_out.Msubunits(Mtar) = nim_swap.subunits(1:NMsubs); % save mult subunits (not offset subunit)
    mnim_out.nim = nim_swap;                                % save upstream/spkNL params
    mnim_out.nim.subunits = mnim.nim.subunits;              % save add subunits
    
    % modify fit history
    mnim_out.nim.fit_props.fit_type = 'Mfilt';
    mnim_out.nim.fit_history(end).fit_type = 'Mfilt';
    
	end

	function mnim_out = fit_alt_filters( mnim, Robs, stims, varargin )
	% Usage: mnim_out = fit_alt_filters( mnim, Robs, stims, varargin )
	%	
	% use flag 'add_first' to fit Afilters before Mfilters
		
		LLtol = 0.0002; MAXiter = 12;

		% Check to see if silent (for alt function)
		[~,parsed_options,modvarargin] = NIM.parse_varargin( varargin, {'silent','add_first'} );
		if isfield( parsed_options, 'silent' )
			silent = parsed_options.silent;
		else
			silent = 0;
		end
				
		modvarargin{end+1} = 'silent';
		modvarargin{end+1} = 1;

		LL = mnim.nim.fit_props.LL; LLpast = -1e10;
		if ~silent
			fprintf( 'Beginning LL = %f\n', LL )
		end
		
		mnim_out = mnim;
		if isfield( parsed_options, 'add_first')
			mnim_out = mnim_out.fit_Afilters( Robs, stims, modvarargin{:} );
		end

    iter = 1;
    while (((LL-LLpast) > LLtol) && (iter < MAXiter))

        mnim_out = mnim_out.fit_Msequential( Robs, stims, modvarargin{:} ); % defaults to fitting filts of all subunits specified in 'subs'; defaults to all
        mnim_out = mnim_out.fit_Afilters( Robs, stims, modvarargin{:} );


        LLpast = LL;
        LL = mnim_out.nim.fit_props.LL;
        iter = iter + 1;

        if ~silent
            fprintf( '  Iter %2d: LL = %f\n', iter, LL )
        end	
    end
    end
	
	function mnim_out = fit_AupstreamNLs( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_AupstreamNLs( Robs, stims, Uindx, varargin )
	
		% Ensure proper format of stims cell array
		if ~iscell(stims)
			tmp = stims;
			clear stims
			stims{1} = tmp;
		end

		% append necessary options to varargin to pass to fit_filters
    varargin{end+1} = 'gain_funs';
		varargin{end+1} = mnim.calc_gmults( stims );
		%varargin{end+1} = 'fit_offsets';
		%varargin{end+1} = 1;

		mnim_out = mnim;
		mnim_out.nim = mnim.nim.fit_upstreamNLs( Robs, stims, varargin{:} );  
	end

 	function mnim_out = fit_MupstreamNLs( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_MupstreamNLs( Robs, stims, Uindx, varargin )
	%
	% Fits upstream nonlinearities of multiplicative subunits
	% Note 1: Specify which Msubunits to optimize using 'subs' option, 
    % numbered by their index in Msubunits array
    % Note 2: method only handles case where the Msubunits to optimize
    % target unique additive subunits; if this is not the case, use
    % fit_Msequential(which will in turn call this method properly)
    
    % ensure proper format of stims cell array
    if ~iscell(stims)
        tmp = stims;
        clear stims
        stims{1} = tmp;
    end

    % parse optional arguments, separating those needed in this method
    % from those needed as inputs to fit_filters method in NIM
    [~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs'} );
    % save indices of subunits targeted for fitting
    if isfield( parsed_inputs, 'subs' )
        Mtar = parsed_inputs.subs;
    else
        Mtar = 1:length( mnim.Msubunits );
    end
    assert(all(ismember(Mtar,1:length(mnim.Msubunits))),'Invalid ''subs'' input')
    NMsubs = length(Mtar);

    % ensure multiple multiplicative units are not acting on the same
    % additive subunit
    Atargs = [];
    for i = 1:NMsubs
        temp_targs = mnim.Mtargets{Mtar(i)};
        Atargs = [Atargs; temp_targs(:)];
    end
    assert(length(Atargs)==length(unique(Atargs)),'Cannot simultaneously fit two mult subunits with same target')

    % swap roles of additive and multiplicative subunits
    [nim_swap,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtar );

    % append necessary options to varargin to pass to fit_filters
    modvarargin{end+1} = 'gain_funs';
    modvarargin{end+1} = gmults;
    modvarargin{end+1} = 'subs';
    modvarargin{end+1} = 1:NMsubs;
    %modvarargin{pos+4} = 'fit_offsets';
    %modvarargin{pos+5} = 1;

    % fit filters using NIM method
    nim_swap = nim_swap.fit_upstreamNLs( Robs, stims_plus, modvarargin{:} );

    % Copy filters back to their locations
    mnim_out = mnim;
    mnim_out.Msubunits(Mtar) = nim_swap.subunits(1:NMsubs); % save mult subunits (not offset subunit)
    mnim_out.nim = nim_swap;                                % save upstream/spkNL params
    mnim_out.nim.subunits = mnim.nim.subunits;              % save add subunits
    
    % modify fit history
    mnim_out.nim.fit_props.fit_type = 'Mupstream_NLs';
    mnim_out.nim.fit_history(end).fit_type = 'Mupstream_NLs';
    
    end
		
	function mnim_out = reg_pathA( mnim, Robs, stims, Uindx, XVindx, varargin )
	% Usage: mnim = reg_path( mnim, Robs, stims, Uindx, XVindx, varargin )

    % ensure proper format of stims cell array
    if ~iscell(stims)
        tmp = stims;
        clear stims
        stims{1} = tmp;
    end

    % append necessary options to varargin to pass to fit_filters
    varargin{end+1} = 'gain_funs';
    varargin{end+1} = mnim.calc_gmults( stims );
    %varargin{end+1} = 'fit_offsets';
    %varargin{end+1} = 1;

    mnim_out = mnim;
    mnim_out.nim = mnim.nim.reg_path( Robs, stims, Uindx, XVindx, varargin{:} );
    end
		
	function mnim_out = reg_pathM( mnim, Robs, stims, Uindx, XVindx, varargin )
	% Usage: mnim = reg_pathM( mnim, Robs, stims, Uindx, XVindx, varargin )

    % parse optional arguments, separating those needed in this method
    % from those needed as inputs to fit_filters method in NIM
    [~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs'} );
    % save indices of subunits targeted for fitting
    if isfield( parsed_inputs, 'subs' )
        Mtar = parsed_inputs.subs;
    else
        Mtar = 1:length( mnim.Msubunits );
    end
    assert(all(ismember(Mtar,1:length(mnim.Msubunits))),'Invalid ''subs'' input')
    NMsubs = length(Mtar);

    % swap roles of additive and multiplicative subunits
    [nim_swap,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtar );

    % append necessary options to varargin to pass to fit_filters
    modvarargin{end+1} = 'gain_funs';
    modvarargin{end+1} = gmults;
    modvarargin{end+1} = 'subs';
    modvarargin{end+1} = 1:NMsubs;

    % use NIM method reg_path
    nim_swap = nim_swap.reg_path( Robs, stims_plus, Uindx, XVindx, modvarargin{:} );

    % Copy filters back to their locations
    mnim_out = mnim;
    mnim_out.Msubunits(Mtar) = nim_swap.subunits(1:NMsubs); % save mult subunits (not offset subunit)
    mnim_out.nim = nim_swap;                                % save upstream/spkNL params
    mnim_out.nim.subunits = mnim.nim.subunits;              % save add subunits
    
    % modify fit history
    mnim_out.nim.fit_props.fit_type = 'Mfilt';
    mnim_out.nim.fit_history(end).fit_type = 'Mfilt';
	end
	    
	function mnim_out = fit_Msequential( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_Msequential( Robs, stims, Uindx, varargin )
	% Fits filters or upstream nonlinearities of multiplicative subunit,
	% and handles case where the Msubunits to optimize target the same 
	% additive subunits
	%
	% INPUTS:
	%   Robs:
	%   stim:
	%   optional flags:
	%       any optional flag argument used for fit_Mfilters method or fit_MupstreamNLs
	%       ('subs',subs):  
	%           option 1 - array of Msubunit indices indicating which
	%               subunits to optimize. If all selected subunits act on
	%               unique additive subunits, fit_Mfilters is called. If
	%               not, the indices are arranged so that fit_Mfilters will
	%               be called sequentially, and in the case of multiple 
	%               Msubunits acting on the same additive subunit, the
	%               Msubunit with the smallest index will by default be fit first.
	%           option 2 - Nx1 cell array of Msubunit indices indicating 
	%               which Msubunits should be fit during N calls to
	%               fit_Mfilters. Indices in each cell of the array will be
	%               checked for uniqueness of targets; if nonuniqueness is
	%               found, that cell is divided into one or more cells and
	%               the rest of the fitting order will be unaffected.
	%           option 3 - if subs is empty or 'subs' is not listed as an
	%               optional flag, all Msubunits are selected and the
	%               method defaults to the behavior in option 1
	%       ('component','component'): string 'filt' or 'upstreamNLs'
	%               specifying which component of the Msubunits to fit;
	%               defaults to 'filt'
	% 
	% OUTPUTS:
	%   mnim_out:   updated multNIM object
    
		% Ensure proper format of stims cell array
		if ~iscell(stims)
			tmp = stims;
			clear stims
			stims{1} = tmp;
		end

    % parse optional arguments, separating those needed in this method
    % from those needed as inputs to fit_filters method in NIM
    [~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs','component'} );

    % parse component flag
    component = 'filt'; % default
    if isfield( parsed_inputs, 'component' )
        component = parsed_inputs.component;
    end
    assert(ismember(component,{'filt','upstreamNLs'}),'Invalid model component specified')

    % save indices of subunits targeted for fitting
    if isfield( parsed_inputs, 'subs' )
        Mtar = parsed_inputs.subs;                           % pull out user-specified Mtar
        % check user input for 'subs'
        if isempty(Mtar)
            Mtar = {1:length(mnim.Msubunits)};               % use all subunits if empty argument
        elseif ~iscell(Mtar)
            Mtar = mat2cell(Mtar,size(Mtar,1),size(Mtar,2)); % if single vector has been input
        end
    else
        Mtar = {1:length(mnim.Msubunits)};                   % use all subunits if flag not set
    end        
    Nfits = length(Mtar);

    % ensure multiple multiplicative units are not acting on the same
    % additive subunit; if they do, split Mtar
    reset_order_flag = 0;
    i = 1; % counter for number of fits
    while ~reset_order_flag && i <= Nfits
        % get additive subunits acted on by current set of subs indices
        Ncurr_Msubs = length(Mtar{i});
        Atargs = [];
        for j = 1:Ncurr_Msubs
            curr_targs = cell2mat(mnim.Mtargets(Mtar{i}(j))); 
            Atargs = [Atargs; curr_targs(:)];
        end
        if length(Atargs)~=length(unique(Atargs))
            warning('Cannot simultaneously fit two mult subunits with same target; changing fit order')
            reset_order_flag = 1;
            break
        end
        i = i+1; % update counter
    end

    % reset order of Mtar if necessary, with lower indices of Msubunits taking
    % priority
    if reset_order_flag
        remaining_Mtargets = unique(cell2mat(Mtar)); % get sorted list of all targeted Msubunits
        Mtar = {};  % reset Mtar
        % loop through targets until none are left
        while ~isempty(remaining_Mtargets)
            curr_Atargs = [];
            rm_indxs = [];
            for i = 1:length(remaining_Mtargets)
                i_Atargs = mnim.Mtargets{remaining_Mtargets(i)};
                if ~any(ismember(curr_Atargs,i_Atargs))
                    curr_Atargs = [curr_Atargs; i_Atargs(:)]; % add targets of current subunit to list
                    rm_indxs = [rm_indxs; i];                 % remove indices of current subunit from remaining targets
                end
            end
            Mtar{end+1} = remaining_Mtargets(rm_indxs);
            remaining_Mtargets(rm_indxs) = [];
        end
        Nfits = length(Mtar); % update Nfits
    end

    % fit model components; change 'subs' option on each iteration
    modvarargin{end+1} = 'subs'; % 'subs' option removed by parse_varargin
    modvarargin{end+1} = [];
    if strcmp(component,'filt')
        for i = 1:Nfits
            modvarargin{end} = Mtar{i};
            mnim = mnim.fit_Mfilters(Robs,stims,modvarargin{:});
            % modify fit history
            mnim.nim.fit_props.fit_type = 'Mfilt';
            mnim.nim.fit_history(end).fit_type = 'Mfilt';
        end
    elseif strcmp(component,'upstreamNLs')
        for i = 1:Nfits
            modvarargin{end} = Mtar{i};
            mnim = mnim.fit_MupstreamNLs(Robs,stims,modvarargin{:});
            % modify fit history
            mnim.nim.fit_props.fit_type = 'Mupstream_NLs';
            mnim.nim.fit_history(end).fit_type = 'Mupstream_NLs';
        end
    end
    mnim_out = mnim;
	end
    
end

%% ********************  Helper Methods ********************************
methods
    
	function mnim = add_Msubunit( mnim, subunit, Mtargets )
	% Usage: mnim = mnim.add_Msubunit( subunit, Mtargets )
	%
	% INPUTS:
	%   subunit:    this can either be a subunit class to add to the NIM 
	%               model, or a number of an existing subunit to clone for 
	%               a multiplicative subunit 
	%   Mtargets:   array specifying subunits that new Msubunit multiplies
	% 
	% OUTPUTS:
	%   mnim:       updated multNIM object with new Msubunit

		% Error checking on inputs
		assert( all(Mtargets <= length(mnim.nim.subunits)), 'Mtargets must be existing subunits.' )

		% check identity of subunit
		if isa(subunit,'double') 
        % clone existing subunit from NIM
        assert( subunit <= length(mnim.nim.subunits), 'subunit to be multiplicative must be a normal subunit' )
        Msubunit = mnim.subunits(subunit);
        mnim.Msubunits(end+1) = Msubunit;
        % Default nonlinearity/scaling?
    else
        mnim.Msubunits = cat(1, mnim.Msubunits, subunit );
    end

    mnim.Mtargets{end+1} = Mtargets;
    
		end
    
	function Msub = clone_to_Msubunit( mnim, subN, weight, filter_flip )
	% Usage: Msub = mnim.clone_to_Msubunit( subN, <weight>, <filter_flip> ) 
	% Uses an existing subunit of an NIM object as an Msubunit
	%
	% INPUTS:
	%   subN:           index of NIM subunit to clone into an Msubunit
	%   <weight>:       desired weight (+/-1) of new subunit
	%   <filter_flip>:  if anything - literally anything - is in this
	%                   input, we'll flip the filter's sign
	% OUTPUTS:
	%   Msub:           new subunit
		
		Nmods = length(mnim.nim.subunits);
		assert( subN <= Nmods, 'Invalid subunit to clone.' )

		Msub = mnim.nim.subunits(subN);
		if (nargin > 2) && ~isempty(weight)
			Msub.weight = weight;
		end
		if nargin > 3
			Msub.filtK = -Msub.filtK;
			if isfield(Msub,'kt')  
				Msub.kt = -Msub.kt;
			end
		end

		% Should reset nonlinearity if parametric or nonparametric and normalize filter (output <1)
		if strcmp( Msub.NLtype, 'nonpar' )
			Msub.NLnonpar.TBy(:) = 0;
		end		
	end

	function [LL, pred_rate, mod_internals, LL_data] = eval_model( mnim, Robs, stims, varargin )
	%	Usage: [LL, pred_rate, mod_internals, LL_data] = mnim.eval_model( Robs, stims, varargin )
	%
	% OUTPUT:
	%   LL:
	%   pred_rate:
	%   mod_internals: struct with following information
	%     gint ...
	
		varargin{end+1} = 'gain_funs';
		gmults = mnim.calc_gmults( stims );
		varargin{end+1} = gmults;

		[LL,pred_rate,mod_internals,LL_data] = mnim.nim.eval_model( Robs, stims, varargin{:} );
		mod_internals.gain_funs = gmults;
		
		% Multiplicative subunit output information
		nimtmp = mnim.nim;
		nimtmp.subunits = mnim.Msubunits;
		[~,~,Mmod_internals] = nimtmp.eval_model( Robs, stims );
		mod_internals.M_gint = Mmod_internals.gint;
		mod_internals.M_fgint = Mmod_internals.fgint;
		
	end
    
	function [sub_outs,fg_add,gmults] = get_subunit_outputs( mnim, stims )
	% Usage: [sub_outs,fg_add,fg_mult] = mnim.get_subunit_outputs( stims )
	% 
	% Calculates output of all subunits (additive and multiplicative) and separates additive and multiplicative components
	% INPUTS:
	%   stims:      cell array of stim matrices for mnim model
	% 
	% OUTPUTS:
	%   sub_outs:   T x num_add_subunits matrix of additive subunit outputs
	%               multiplied by their corresponding gain functions
	%   fg_add:     T x num_add_subunits matrix of additive subunit outputs
	%   fg_mult:    T x num_add_subunits matrix of gmults from Msubunit outputs
	
		% Get additive subunit outputs
		[~,~,mod_internals] = mnim.nim.eval_model( [], stims );
		fg_add = mod_internals.fgint;

		% Multiply by excitatory and inhibitory weights
		for nn = 1:length(mnim.nim.subunits)
			fg_add(:,nn) = fg_add(:,nn) * mnim.nim.subunits(nn).weight;
		end
	
		% Calculate multiplicative effect
		gmults = mnim.calc_gmults(stims);  % T x num_add_subunits; corresponds to fg_add, contain 1+f()  
		sub_outs = fg_add;                 % default value is additive subunit output
		for nn = 1:length(mnim.Msubunits)
			sub_outs(:,mnim.Mtargets{nn}) = fg_add(:,mnim.Mtargets{nn}) .* gmults(:,mnim.Mtargets{nn}); 
		end
	end
 		
	function mnim = init_nonpar_NLs( mnim, stims, varargin )
	% Usage: mnim = mnim.init_nonpar_NLs( stims, varargin )
	% Initializes the specified model subunits to have nonparametric (tent-basis) upstream NLs,
	% inherited from NIM version. 
	%
	% Note: default is initializes all subunits and Msubunits. Change this through optional flags
	%
	% INPUTS: 
	%   stims: cell array of stimuli
	%   optional flags:
	%       ('subs',sub_inds): Index values of set of subunits to make nonpar (default is all)
	%       ('Msubs',sub_inds): Index values of set of Msubunits to make nonpar (default is all)
	%       ('lambda_nld2',lambda_nld2): specify strength of smoothness regularization for the tent-basis coefs
	%       ('NLmon',NLmon): Set to +1 to constrain NL coefs to be monotonic increasing and
	%						 -1 to make monotonic decreasing. 0 means no constraint. Default here is +1 (monotonic increasing)
	%		('edge_p',edge_p): Scalar that determines the locations of the outermost tent-bases 
	%                       relative to the underlying generating distribution
	%       ('n_bfs',n_bfs): Number of tent-basis functions to use 
	%       ('space_type',space_type): Use either 'equispace' for uniform bin spacing, or 'equipop' for 'equipopulated bins' 
	% OUTPUTS: 
	%   mnim: new mnim object
		
		% Initialize subunit NLs
		[~,parsed_options,modvarargin] = NIM.parse_varargin( varargin, {'Msubs'} );
		mnim.nim = mnim.nim.init_nonpar_NLs( stims, modvarargin{:} );

		% Initialize Msubunit NLs
		if isfield(parsed_options,'Msubs') 
			Msubs = parsed_options.Msubs;
		else
			Msubs = 1:length(mnim.Msubunits);
		end
    if ~isempty(Msubs)
        [~,~,modvarargin] = NIM.parse_varargin( modvarargin, {'subs'} );
        modvarargin{end+1} = 'subs';
        modvarargin{end+1} = Msubs;
        nimtmp = mnim.nim;
        nimtmp.subunits = mnim.Msubunits;
        nimtmp = nimtmp.init_nonpar_NLs( stims, modvarargin{:} );
        mnim.Msubunits = nimtmp.subunits;
    end	
    
	end
		
	function display_model( mnim, stims, Robs )
	% Usage: [] = mnim.display_model( Xstim, Robs )
	% Model display function that plots additive subunits in left column and multiplicative in right column
	% Tis a simple display, and could be elaborated...
	
		if nargin < 3, Robs = [];	end
		if nargin < 2
			mod_outs = [];
		else
			[~,~,mod_outs] = mnim.eval_model( Robs, stims );
		end
		
		extra_plots = [(mnim.nim.spk_hist.spkhstlen > 0) ~isempty(mod_outs)]; % will be spike-history or spkNL plot?

		NAmods = length(mnim.nim.subunits);
		NMmods = length(mnim.Msubunits);
		
		Nrows = max([NAmods NMmods]);
		if sum(extra_plots) == 0
			Ncols = 6;
		else
			% Then need extra column (and possibly extra row)
			Nrows = max([Nrows sum(extra_plots)]);
			Ncols = 7;
		end

		figure
		% Plot Additive subunits
		for nn = 1:NAmods
			dims = mnim.nim.stim_params(mnim.nim.subunits(nn).Xtarg).dims;
			mnim.nim.subunits(nn).display_filter( dims, [Nrows Ncols (nn-1)*Ncols+1], 'notitle', 1 );
			subplot( Nrows, Ncols, (nn-1)*Ncols+3 )
			if isempty(mod_outs)
				mnim.nim.subunits(nn).display_NL();
			else
				mnim.nim.subunits(nn).display_NL( mod_outs.gint(:,nn) );
			end
			subplot( Nrows, Ncols, (nn-1)*Ncols+1 )
			if mnim.nim.subunits(nn).weight > 0
				title( sprintf( 'Exc #%d', nn ) )
			else
				title( sprintf( 'Sup #%d', nn ) )
			end
		end

		% Plot Mult subunits
		for nn = 1:NMmods
			dims = mnim.nim.stim_params(mnim.nim.subunits(nn).Xtarg).dims;
			mnim.Msubunits(nn).display_filter( dims, [Nrows Ncols (nn-1)*Ncols+4], 'notitle', 1 );
			subplot( Nrows, Ncols, (nn-1)*Ncols+6 )
			if isempty(mod_outs)
				mnim.Msubunits(nn).display_NL( 'y_offset',1.0,'sign',1 );
			else
				mnim.Msubunits(nn).display_NL( mod_outs.M_gint(:,nn), 'y_offset',1.0,'sign',1 );
			end
			subplot( Nrows, Ncols, (nn-1)*Ncols+4 )
			title( sprintf( 'Mult #%d -> %d', nn, mnim.Mtargets{nn}(1) ) )  % at the moment only displays first Mtarget
		end

		% Plot spkNL
		if sum(extra_plots) == 0
			return
		end
		subplot( Nrows, Ncols, Ncols );
		if extra_plots(2) > 0
			mnim.nim.display_spkNL( mod_outs.G );
			title( 'Spk NL' )
			if extra_plots(1) > 0
				subplot( Nrows, Ncols, 2*Ncols );
			end
		end
		if extra_plots(1) > 0
			mnim.nim.display_spike_history();
		end
	end
	
	function display_model_old( mnim, Xstim, Robs )
	% Usage: [] = mnim.display_model( Xstim, Robs )
		  
		if nargin < 2, Xstim = []; end
		if nargin < 3, Robs = []; end
	
		nimtmp = mnim.nim;
		Nmods = length(nimtmp.subunits);
		fprintf( 'Regular Subunits: 1-%d\n   Mult Subunits: %d-%d\n', Nmods, Nmods+1, Nmods+length(mnim.Msubunits) )
		nimtmp.subunits = cat(1,nimtmp.subunits(:), mnim.Msubunits(:) );
		if strcmp(class(nimtmp),'NIM')
			nimtmp.display_model_dab( Xstim, Robs );
		else
			nimtmp.display_model( Xstim, Robs );  
		end    
	end

end

%% ********************  hidden methods ********************************
methods (Hidden)
	
	function [gmults, fg_mult] = calc_gmults( mnim, stims )
	% Usage: gmults = mnim.calc_gmults( stims )
	% Hidden method used to calculate gains that will then be passed to an NIM fitting method or eval method
	% 
	% INPUTS:
	%   stims:      cell array of stim matrices for model fitting
	% OUTPUTS:
	%   gmults:     T x num_add_units matrix of gmults, one for each additive subunit in mnim.nim NIM object
	%   fg_mults:   T x num_mult_units matrix of multiplicative subunit outputs (includes weights)
			
		% set default gmults
		[~,~,mod_int_add] = mnim.nim.eval_model( [], stims ); % cheap way to get size of gmults for diff nim classes
		gmults = ones(size(mod_int_add.gint));  % default gains of 1 for every subunit 

		% change defaults
		if ~isempty(mnim.Msubunits)
			% create standard NIM model whose subunits come from Msubunits for easy evaluation of subunit outputs
			nimtmp = mnim.nim;
			nimtmp.subunits = mnim.Msubunits;
			[~,~,mod_int_mult] = nimtmp.eval_model( [], stims );
			fg_mult = mod_int_mult.fgint;
			% only change gmults for additive subunits targeted by Msubunits
			for nn = 1:length(mnim.Msubunits)
				% multiply existing gmults of targets by output of selected subunit
				targets = mnim.Mtargets{nn};
				weight = mnim.Msubunits(nn).weight;
				fg_mult(:,nn) = weight*fg_mult(:,nn);
				gmults(:,targets) = gmults(:,targets).*repmat(1+fg_mult(:,nn),1,length(targets));
			end
		end	
	end
	
	function [nim,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtargs )
	% Usage: [nim,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtargs )
	% 
	% setup for fitting multiplicative subunits as NIM additive units with 
	% gmults taking care of outputs from original additive subunits 
	% original multiplicative subunits that are not being fit. Also outputs
	% proper stim cell array.
	% Note: method is called from fit_Mfilters, so no two multiplicative 
	% subunits specified by Mtargets should have the same additive subunit
	% as a target (though a given Mtarget can have multiple additive subunits as targets)
	%
	% INPUTS:
	%   mnim:       multNIM object whose multiplicative subunits are being fit
	%   stims:      cell array of stim matrices for original additive and
	%               multiplicative subunits
	%   Mtargs:     array of indices specifying which multiplicative
	%               subunits are to be fit
	% OUTPUTS:
	%   nim:        updated NIM object that contains subunits to be fit
	%   gmults:     T x num(Mtargets) matrix of gmult values obtained from
	%               original subunits and nontarget multiplicative subunits
	%   stims_plus: same cell array as stims with an additional stim matrix
	%               to properly take into account offsets due to shuffling around model terms
	
		Mnontargs = setdiff(1:length(mnim.Msubunits),Mtargs); % list of nontargeted mult subunits

		% Ensure proper format of stims cell array
		if ~iscell(stims)
			stims_plus{1} = stims;
		else
			stims_plus = stims;
		end

		% Extract relevant outputs from add and mult subunits
		[~,fg_add] = mnim.get_subunit_outputs( stims_plus ); % fg_add incorporates weight
		[~,fg_mult] = mnim.calc_gmults( stims_plus );        % fg_mult incorporates weight

		% Multiply output of additive subunits and non-target mult subunits
		g = fg_add;
		for i = Mnontargs
			targs = mnim.Mtargets{i};
			g(:,targs) = g(:,targs).*repmat((1+fg_mult(:,i)),1,length(targs));
		end

		% Now each Mtarg should be multiplied by the g's corresponding to its targets
		NMsubs = length(Mtargs);
		gmults = ones(size(g,1),NMsubs+1);  % extra-dim of ones is to multiply additive offset term
		for i = 1:length(Mtargs)
			gmults(:,i) = sum(g(:,mnim.Mtargets{Mtargs(i)}),2);
		end
    % gmults for the offset is given by sum of all g's, since those
    % associated with targets being fit have a +1 and those not still
    % need to be included
    gmults(:,end) = sum(g,2);

    % add "stimulus" for offset term to stims  
    offset_xtarg = length(stims_plus)+1;
    stims_plus{offset_xtarg} = ones(size(g,1),1);

    % Construct NIM for filter minimization
    nim = mnim.nim;
    nim.subunits = mnim.Msubunits(Mtargs);

    % make last NIM-subunit add in additive terms
    nim.subunits(NMsubs+1) = nim.subunits(NMsubs);
    nim.subunits(end).Xtarg = offset_xtarg;
    nim.subunits(end).filtK = 1;
    nim.subunits(end).NLoffset = 0;
    if isa(nim.subunits(end),'LRSUBUNIT')
        nim.subunits(end).kt = 1;
        nim.subunits(end).ksp = 1;
    end
    nim.subunits(end).NLtype = 'lin';
    nim.subunits(end).weight = 1;
    nim.subunits(end).reg_lambdas = SUBUNIT.init_reg_lambdas();

    % Add offset terms to stim_params
    stimpar1 = nim.stim_params(1);
    stimpar1.dims = [1 1 1];
    stimpar1.tent_spacing = [];
    stimpar1.up_fac = 1;  % since any up_fac is already taken into account in subunit outputs
    nim.stim_params(offset_xtarg) = stimpar1;
				
    end
	
end

end

