classdef multNIM
% Class implementation of multiplicative NIM based on NIM class. 

properties
	nim;			% NIM or sNIM struct that contains the normal subunits
	Msubunits;		% struct of multiplicative subunits with the following fields:
		% subunit	% actual NIM/sNIM subunits that should multiply Mtargets
		% targets;	% array of targets in the nim
		% weights;	% array of weights on the output of the Msubunits (one for each target)
end	

properties (Hidden)
	version = '0.4';    %source code version used to generate the model
	create_on = date;   %date model was generated
end	

%% ******************** constructor ********************************
methods

	function mnim = multNIM( nim, Msubunits, Mtargets, Mweights )
	% Usage: mnim = multNIM( nim, <Msubunits, Mtargets>, <Mweights> )
	%
	% INPUTS:
	%   nim:        either an object of the NIM or sNIM class
	%   Msubunits:  array of SUBUNIT objects that will act as multiplicative
	%               gains on the subunits of the nim object
	%   Mtargets:   array of integers if each subunit only targets a single
	%               additive subunit, or a cell array, where each cell
	%               contains the targets of the corresponding Msubunit
	%	<Mweights>:	array of integers if each subunit only targets a single
	%               additive subunit, or a cell array, where each cell
	%               contains the weights that multiply the output of the 
	%				Msubunit before adding 1 for each target;
	%				one weight for each target of the subunit, or else
	%				defaults to all ones
	%
	% OUTPUTS:
	%   mnim:       initialized multNIM object
	
		% Handle the no-input-argument case by returning a null model. This is important when initializing arrays of objects
		if nargin < 3
			if nargin > 0
				mnim.nim = nim;
			else
				mnim.nim = [];
			end
			mnim.Msubunits = [];
			return
		end

		% Error checking on inputs
		assert( nargin >= 3, 'Must specify targets as well as subunits' )
		
		NMsubs = length(Msubunits);
		
		% Define defaults
		mnim.nim = nim;
		mnim.Msubunits(1).subunit = [];
		mnim.Msubunits(1).targets = [];
		mnim.Msubunits(1).weights = [];

		% Turn Mtargets into a cell array if not already
		if ~iscell(Mtargets)
			if NMsubs == 1
				temp = Mtargets;
				clear Mtargets
				Mtargets{1} = temp;				% if only 1 subunit specified and Mtargets is an array, assume all are targets
			else
				Mtargets = num2cell(Mtargets);	% if more than 1 subunit specified, assume a 1-1 correspondence
				assert( length(Mtargets) == NMsubs, 'Must specify at least 1 target per Msubunit')
			end
		end

		% error checking on Mtargets
		assert( all(cellfun(@(x) all(ismember(x,1:length(nim.subunits))),Mtargets)), 'Invalid Mtargets.' )
		
		% error checking/default setting on Mweights
		if nargin == 4
			% Turn Mweights into a cell array if not already
			if ~iscell(Mweights)
				Mweights = num2cell(Mweights);
			end
			for i = 1:NMsubs
				assert( length(Mtargets{i}) == length(Mweights{i}) );	% make sure number of targets == number of weights
			end
		else
			for i = 1:NMsubs
				Mweights{i} = ones(length(Mtargets{i}),1);				% default to weights of 1
			end
		end
		
		
		% set multNIM properties
		mnim.nim = nim;
		mnim.Msubunits = struct([]);	% start with empty struct
		for i = 1:NMsubs
			mnim.Msubunits(end+1).subunit = Msubunits(i);
			mnim.Msubunits(end).targets = Mtargets{i}(:)';		% make row vector for easy manipulation later
			mnim.Msubunits(end).weights = Mweights{i}(:)';		% make row vector for easy manipulation later
		end
		
	end
	
end

%% ******************** Fitting methods ********************************
methods
	
	function mnim_out = fit_Afilters( mnim, Robs, stims, varargin )
	% Usage: mnim = mnim.fit_Afilters( Robs, stims, Uindx, varargin )
	% Fits filters of additive subunits
    
		% Ensure proper format of stims cell array
		if ~iscell(stims)
			tmp = stims;
			clear stims
			stims{1} = tmp;
		end

		% Append necessary options to varargin to pass to fit_filters
		varargin{end+1} = 'gain_funs';
		varargin{end+1} = mnim.calc_gmults( stims );

		% Fit filters
		mnim_out = mnim;
		mnim_out.nim = mnim.nim.fit_filters( Robs, stims, varargin{:} );    
	end
	
	function mnim_out = fit_Mfilters( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_Mfilters( Robs, stims, Uindx, varargin )
	%
	% Fits filters of multiplicative subunit
	% Note 1: Specify which Msubunits to optimize using 'subs' option, numbered by their index in Msubunits struct array
	% Note 2: method only handles case where the Msubunits to optimize target unique additive subunits; if this is
	% not the case, use fit_Msequential (which will in turn call this method properly)
    
		% Ensure proper format of stims cell array
		if ~iscell(stims)
			tmp = stims;
			clear stims
			stims{1} = tmp;
		end

		% Parse optional arguments, separating those needed in this method from those needed as inputs to fit_filters method in NIM
		[~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs'} );
		% Save indices of subunits targeted for fitting
		if isfield( parsed_inputs, 'subs' ) 
			Mtar = parsed_inputs.subs;
		else
			Mtar = 1:length( mnim.Msubunits );	% default behavior: fit all Msubunits
		end
		assert(all(ismember(Mtar,1:length(mnim.Msubunits))),'Invalid ''subs'' input')
		NMsubs = length(Mtar);

		% Ensure multiple multiplicative units are not acting on the same additive subunit
		Atargs = mnim.Msubunits(Mtar).targets;	% concatenated targets from all subunits in Mtar; targets need to be row vecs 
		assert(length(Atargs)==length(unique(Atargs)),'Cannot simultaneously fit two mult subunits with same target; use fit_Msequential method')
		
		% Swap roles of additive and multiplicative subunits
		[nim_swap,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtar );

		% Append necessary options to varargin to pass to fit_filters
		modvarargin{end+1} = 'gain_funs';
		modvarargin{end+1} = gmults;
		modvarargin{end+1} = 'subs';
		modvarargin{end+1} = 1:NMsubs;

		% Fit filters using NIM method
		nim_swap = nim_swap.fit_filters( Robs, stims_plus, modvarargin{:} );

		% Copy filters back to their locations
		mnim_out = mnim;
		for nn = 1:NMsubs  % save mult subunits (not offset subunit)
			mnim_out.Msubunits(Mtar(nn)).subunit = nim_swap.subunits(nn);	
		end
		mnim_out.nim = nim_swap;											% save upstream/spkNL params
		mnim_out.nim.subunits = mnim.nim.subunits;							% save add subunits
    
		% Modify fit history
		mnim_out.nim.fit_props.fit_type = 'Mfilt';
		mnim_out.nim.fit_history(end).fit_type = 'Mfilt';
    
	end

	function mnim_out = fit_filters( mnim, Robs, stims, varargin )
	% Use either fit_Afilters or fit_Mfilters/fit_Msequential (or fit_alt_filters)
		warning( 'Use either multNIM.fit_Afilters or fit_Mfilters/fit_Msequential (or fit_alt_filters). Defaulting to fit_Afilters.' )
		mnim_out = mnim.fit_Afilters( Robs, stims, varargin{:} );
	end
	
	function mnim_out = fit_Aweights( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_Aweights( Robs, stims, varargin )
	% Fits weights on additive subunits (given fixed multiplicative modulation)

		varargin{end+1} = 'gain_funs';
		gmults = mnim.calc_gmults( stims );
		varargin{end+1} = gmults;

		mnim_out = mnim;
		mnim_out.nim = mnim.nim.fit_weights( Robs, stims, varargin{:} );
	end
	
	function mnim_out = fit_Mweights( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_Mweights( Robs, stims, Uindx, varargin )
	%
	% Fits weights w_{ij} of multiplicative subunit i on additive subunit
	% j, where the gain signal acting on additive subunit j is 
	% (1 + w_{i,j}*(output of mult subunit i)
	%
	% Note 1: Specify which Msubunit weights to optimize using 'subs' option, 
	%	numbered by their index in Msubunits array
	% Note 2: Specify which weights from the chosen Msubunits to fit using the 
	%	'weight_targs' option. Should be a cell array whose ith cell
	%	contains a subset of the values in 'targets' field from the ith 
	%	Msubunit in the 'subs' array
	% Note 3: method only handles case where at most one weight per
	%	additive subunit is fit. In the event that more than one weight is
	%	specified the method will exit with an error message.
	
		% Ensure proper format of stims cell array
		if ~iscell(stims)
			tmp = stims;
			clear stims
			stims{1} = tmp;
		end

		% Parse optional arguments, separating those needed in this method from those needed as inputs to fit_filters method in NIM
		[~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs','weight_targs'} );
		
		% Save indices of mult subunits targeted for fitting
		if isfield( parsed_inputs, 'subs' ) 
			Mtar = parsed_inputs.subs;
		else
			Mtar = 1:length( mnim.Msubunits );
		end
		assert(all(ismember(Mtar,1:length(mnim.Msubunits))),'Invalid ''subs'' input')
		NMsubs = length(Mtar);

		% Save indices of additive subunits targeted for fitting
		if isfield( parsed_inputs, 'weight_targs' ) 
			Atar = parsed_inputs.weight_targs;
		else
			% default is all targets of each Mtar
 			[Atar{1:NMsubs}] = mnim.Msubunits(Mtar).targets;	% each cell of Atar will contain targets from one Msubunit
		end
		
		% Ensure multiple weights are not being fit on a single additive
		% subunit
		assert(length([Atar{:}])==length(unique([Atar{:}])),'Cannot simultaneously fit two weights acting on same additive subunit')
		% Ensure specified Atars match up with possible Msubunit targets
		for i = 1:length(Mtar)
			assert(all( ismember(Atar{i}, mnim.Msubunits(Mtar(i)).targets)), 'Atar does not match possible targets for Msubunit %i',i);
		end
		
		% Creat NIM for fitting weights; a single subunit will contain a
		% filter with all the combined weights that are being fit
		[nim_weight,stims_plus] = format4Mweightfitting( mnim, stims, Mtar, Atar );

		% Append necessary options to varargin to pass to fit_filters
		modvarargin{end+1} = 'subs';
		modvarargin{end+1} = 1;			% just fit weights

		% Fit filters using NIM method
		nim_weight = nim_weight.fit_filters( Robs, stims_plus, modvarargin{:} );

		% Copy filters back to their locations
		mnim_out = mnim;
		% Msubunits have not changed
		mnim_out.nim = nim_weight;                              % to save upstream/spkNL params
		mnim_out.nim.stim_params = mnim.nim.stim_params;		% to save stim_params
		mnim_out.nim.subunits = mnim.nim.subunits;              % to save additive subunits
		
		% update Mweights, which are found in nim_swap filter
		beg_indx = 0;
		for i = 1:NMsubs
			[~,Atar_indx] = ismember(Atar{i},mnim_out.Msubunits(Mtar(i)).targets); % get indices of updated weights
			mnim_out.Msubunits(Mtar(i)).weights(Atar_indx) = nim_weight.subunits(1).filtK(beg_indx+(1:length(Atar{i})))'; % filtK is column vec
			beg_indx = beg_indx+length(Atar{i});
		end
    
		% Modify fit history
		mnim_out.nim.fit_props.fit_type = 'Mweight';
		mnim_out.nim.fit_history(end).fit_type = 'Mweight';
    
	end
	
	function mnim_out = fit_weights( mnim, Robs, stims, varargin )
	% Use either fit_Aweights or fit_Mweights/fit_Msequential (or fit_alt_filters)
		warning( 'Use either multNIM.fit_Aweights or fit_Mweights. Defaulting to fit_Aweights.' )
		mnim_out = mnim.fit_Aweights( Robs, stims, varargin{:} );
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
	Atar = [mnim.Msubunits(Mtar).targets];
    assert(length(Atar)==length(unique(Atar)),'Cannot simultaneously fit two mult subunits with same target')

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
	[mnim_out.Msubunits(Mtar).subunit] = nim_swap.subunits(1:NMsubs);	% save mult subunits (not offset subunit)
    mnim_out.nim = nim_swap;											% save upstream/spkNL params
    mnim_out.nim.subunits = mnim.nim.subunits;							% save add subunits
    
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
    [mnim_out.Msubunits(Mtar).subunit] = nim_swap.subunits(1:NMsubs);	% save mult subunits (not offset subunit)
    mnim_out.nim = nim_swap;											% save upstream/spkNL params
    mnim_out.nim.subunits = mnim.nim.subunits;							% save add subunits
    
    % modify fit history
    mnim_out.nim.fit_props.fit_type = 'Mfilt';
    mnim_out.nim.fit_history(end).fit_type = 'Mfilt';
	end
	    
	function mnim_out = fit_Msequential( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_Msequential( Robs, stims, Uindx, varargin )
	% Fits filters or upstream nonlinearities of multiplicative subunit, and handles case 
	% where the Msubunits to optimize target the same additive subunits
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

		% Parse optional arguments, separating those needed in this method from those needed as inputs to fit_filters method in NIM
		[~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs','component'} );

		% Parse component flag
		component = 'filt'; % default
		if isfield( parsed_inputs, 'component' )
			component = parsed_inputs.component;
		end
		assert(ismember(component,{'filt','upstreamNLs'}),'Invalid model component specified')

		% Save indices of subunits targeted for fitting
		if isfield( parsed_inputs, 'subs' )
			Mtar = parsed_inputs.subs;								% pull out user-specified Mtar
			% check user input for 'subs'
			if isempty(Mtar)
				Mtar = {1:length(mnim.Msubunits)};					% use all subunits if empty argument
			elseif ~iscell(Mtar)
				Mtar = mat2cell(Mtar,size(Mtar,1),size(Mtar,2));	% if single vector has been input, convert to cell array
			end
		else
			Mtar = {1:length(mnim.Msubunits)};						% use all subunits if flag not set
		end        
		Nfits = length(Mtar);	% total number of fits; Mtar is cell array, so Nfits might equal 1 even if fitting multiple Msubunits

		% Ensure multiple multiplicative units are not acting on the same additive subunit; if they do, split Mtar
		reset_order_flag = 0;
		i = 1; % counter for number of fits
		while ~reset_order_flag && i <= Nfits
			% get additive subunits acted on by current set of subs indices
			Atar = [mnim.Msubunits(Mtar{i}).targets];
			if length(Atar)~=length(unique(Atar))
				warning('Cannot simultaneously fit two mult subunits with same target; changing fit order')
				reset_order_flag = 1;
				break
			end
			i = i+1; % update counter
		end

		% Reset order of Mtar if necessary, with lower indices of Msubunits taking priority
		if reset_order_flag
			remaining_Mtargets = unique([Mtar{:}]); % get sorted list of all targeted Msubunits
			Mtar = {};  % reset Mtar
			% Loop through targets until none are left
			while ~isempty(remaining_Mtargets)
				curr_Atargs = [];
				rm_indxs = [];
				for i = 1:length(remaining_Mtargets)
					i_Atargs = mnim.Msubunits(remaining_Mtargets(i)).targets;
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

		% Fit model components; change 'subs' option on each iteration
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
    
	function mnim = add_Msubunit( mnim, subunit, targets, weights )
	% Usage: mnim = mnim.add_Msubunit( subunit, targets, <weights>)
	%
	% INPUTS:
	%   subunit:    this can either be a subunit class to add to the NIM 
	%               model, or a number of an existing subunit to clone for 
	%               a multiplicative subunit 
	%   targets:	array specifying additive subunits that new Msubunit multiplies
	%	weights:	array specifying weights on subunit output before 1 is
	%				added; defaults to a weight vector of 1s
	% 
	% OUTPUTS:
	%   mnim:       updated multNIM object with new Msubunit

		% Error checking on inputs
		assert( all(targets <= length(mnim.nim.subunits)), 'Mtargets must be existing subunits.' )

		if nargin < 4
			weights = ones(1,length(targets));
		end
		
		% check identity of subunit
		if isa(subunit,'double') 
			% clone existing subunit from NIM
			assert( subunit <= length(mnim.nim.subunits), 'subunit to be multiplicative must be a normal subunit' )
			mnim.Msubunits(end+1).subunit = mnim.subunits(subunit);
			% Default nonlinearity/scaling?
		else
			mnim.Msubunits(end+1).subunit = subunit;
		end

		mnim.Msubunits(end).targets = targets;
		mnim.Msubunits(end).weights = weights;
    
	end
    
	function Msub = clone_to_Msubunit( mnim, subN, weight, filter_flip )
	% Usage: Msub = mnim.clone_to_Msubunit( subN, <weight>, <filter_flip> ) 
	% Uses an existing subunit of an NIM object as an Msubunit
	%
	% INPUTS:
	%   subN:           index of NIM subunit to clone into an Msubunit
	%   <weight>:	    desired weight (+/-1) of new subunit; this is
	%					different from Msubunits(_).weights, which weight the
	%					output of this subunit differently for each of its
	%					targets
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
		nimtmp.subunits = [mnim.Msubunits(:).subunit];
		[~,~,Mmod_internals] = nimtmp.eval_model( Robs, stims );
		mod_internals.M_gint = Mmod_internals.gint;
		mod_internals.M_fgint = Mmod_internals.fgint; % Note that these are not weighted by Mweights!
		
	end
    
	function [sub_outs,fg_add,gmults] = get_subunit_outputs( mnim, stims )
	% Usage: [sub_outs,fg_add,gmults] = mnim.get_subunit_outputs( stims )
	% 
	% Calculates output of all subunits (additive and multiplicative) as
	% well as separate additive and multiplicative components; called by
	% format4Mfitting
	% INPUTS:
	%   stims:      cell array of stim matrices for mnim model
	% 
	% OUTPUTS:
	%   sub_outs:   T x num_add_subunits matrix of additive subunit outputs
	%					multiplied by their corresponding gain functions
	%   fg_add:     T x num_add_subunits matrix of additive subunit outputs
	%					(includes multiplication by subunit weight)
	%   gmults:     T x num_add_subunits matrix of gmults from Msubunit outputs
	%					
		% Get additive subunit outputs
		[~,~,mod_internals] = mnim.nim.eval_model( [], stims );
		fg_add = mod_internals.fgint;

		% Multiply by excitatory and inhibitory weights
		for nn = 1:length(mnim.nim.subunits)
			fg_add(:,nn) = mnim.nim.subunits(nn).weight * fg_add(:,nn);
		end
	
		% Calculate multiplicative gains
		gmults = mnim.calc_gmults(stims);  % T x num_add_subunits; corresponds to fg_add, contains prod_i( 1+w_i*f() )
		sub_outs = fg_add;                 % default value is additive subunit output
		for nn = 1:length(mnim.Msubunits)
			sub_outs(:,mnim.Msubunits(nn).targets) = fg_add(:,mnim.Msubunits(nn).targets) .* gmults(:,mnim.Msubunits(nn).targets); 
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
			nimtmp.subunits = [mnim.Msubunits(:).subunit];
			nimtmp = nimtmp.init_nonpar_NLs( stims, modvarargin{:} );
			[mnim.Msubunits(:).subunit] = nimtmp.subunits;
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
			dims = mnim.nim.stim_params(mnim.Msubunits(nn).subunit.Xtarg).dims;
			mnim.Msubunits(nn).subunit.display_filter( dims, [Nrows Ncols (nn-1)*Ncols+4], 'notitle', 1 );
			subplot( Nrows, Ncols, (nn-1)*Ncols+6 )
			if isempty(mod_outs)
				mnim.Msubunits(nn).subunit.display_NL( 'y_offset',1.0,'sign',1 );
			else
				mnim.Msubunits(nn).subunit.display_NL( mod_outs.M_gint(:,nn), 'y_offset',1.0,'sign',1 );
			end
			subplot( Nrows, Ncols, (nn-1)*Ncols+4 )
			title( sprintf( 'Mult #%d -> %d', nn, mnim.Msubunits(nn).targets(1) ) )  % at the moment only displays first Mtarget
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
		nimtmp.subunits = cat(1,nimtmp.subunits(:), [mnim.Msubunits(:).subunit]' );
		if strcmp(class(nimtmp),'NIM')
			nimtmp.display_model_dab( Xstim, Robs );
		else
			nimtmp.display_model( Xstim, Robs );  
		end    
	end

end

%% ********************  hidden methods ********************************
methods (Hidden)
	
	function [gmults, fg_xmult, fg_mult] = calc_gmults( mnim, stims )
	% Usage: gmults = mnim.calc_gmults( stims )
	% Hidden method used to calculate gains that will then be passed to an NIM fitting method or eval method
	% 
	% INPUTS:
	%   stims:      cell array of stim matrices for model fitting
	% OUTPUTS:
	%   gmults:     T x num_add_subunits matrix of gmults, one for each 
	%				additive subunit in mnim.nim NIM object
	%				used by get_subunit_output and format4Mfitting
	%   fg_xmult:   "Extended" fg_mult;
	%				1 x num_Msubunits cell array of multiplicative subunit 
	%				outputs. Cell i contains the output of Msubunit(i) for 
	%				ALL its targets (a T x num_targs matrix), since these outputs
	%				may be different depending on the weights (col i is
	%				w_i*f() )
	%				% used by format4Mfitting
	%	fg_mult:	T x num_Msubunits matrix of Msubunit outputs
	%				(includes multiplication by subunit weight)
	%				% used by format4Mweightfitting
	
		% set default gmults
		[~,~,mod_int_add] = mnim.nim.eval_model( [], stims );	% cheap way to get size of gmults for diff nim classes
		gmults = ones(size(mod_int_add.gint));					% default gains of 1 for every subunit 

		% change defaults
		if ~isempty(mnim.Msubunits)
			% create standard NIM model whose subunits come from Msubunits for easy evaluation of subunit outputs
			nimtmp = mnim.nim;
			nimtmp.subunits = [mnim.Msubunits(:).subunit];
			[~,~,mod_int_mult] = nimtmp.eval_model( [], stims );
			
			fg_mult  = zeros(size(gmults,1),length(mnim.Msubunits));
			fg_xmult = cell(1,length(mnim.Msubunits));
			% for each targeted additive subunit in Msubunits.targets, weight Msubunit outputs by weights in
			% subunits as well as weights in Msubunits.weights, and add 1
			for nn = 1:length(mnim.Msubunits)
				targets = mnim.Msubunits(nn).targets;
				weights = mnim.Msubunits(nn).weights;
				% multiply by subunit weight (+/-1)
				fg_mult(:,nn) = mnim.Msubunits(nn).subunit.weight*mod_int_mult.fgint(:,nn);
				% multiply by weights specified in Msubunits.weights and
				% add 1
				fg_xmult{nn} = bsxfun(@times,repmat(fg_mult(:,nn),1,length(targets)),weights(:)');
				gmults(:,targets) = gmults(:,targets).*(1+fg_xmult{nn});
			end
		end	
	end
	
	function [nim,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtar )
	% Usage: [nim,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtar )
	% 
	% setup for fitting multiplicative subunits as NIM additive units with 
	% gmults taking care of outputs from original additive subunits and 
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
	%   Mtar:		array of indices specifying which multiplicative
	%               subunits are to be fit
	%
	% OUTPUTS:
	%   nim:        updated NIM object that contains subunits to be fit
	%   gmults:     T x num_Mtargs matrix of gmult values obtained from
	%               additive subunits and nontarget multiplicative subunits
	%   stims_plus: same cell array as stims with an additional stim matrix
	%               to properly take into account offsets due to shuffling around model terms
	
		Mnontargs = setdiff(1:length(mnim.Msubunits),Mtar); % list of nontargeted mult subunits

		% Ensure proper format of stims cell array
		if ~iscell(stims)
			stims_plus{1} = stims;
		else
			stims_plus = stims;
		end

		% Extract relevant outputs from add and mult subunits
		[~,fg_add] = mnim.get_subunit_outputs( stims ); % fg_add incorporates weight
		[~,fg_xmult] = mnim.calc_gmults( stims );		% fg_xmult incorporates weights (from both the NIM 'weight' property and Mweights) but not 1+ offset

		% Multiply output of additive subunits and non-target mult subunits
		g = fg_add;
		for i = Mnontargs
			targs = mnim.Msubunits(i).targets;
			g(:,targs) = g(:,targs).*(1+fg_xmult{i});
		end

		% Now each Mtarg should be multiplied by the weighted sum of the g's corresponding to
		% its targets
		NMsubs = length(Mtar);
		gmults = ones(size(g,1),NMsubs+1);  % extra-dim of ones is to multiply additive offset term of all 1's
		for i = 1:length(Mtar)
			gmults(:,i) = sum(bsxfun(@times,g(:,mnim.Msubunits(Mtar(i)).targets),mnim.Msubunits(Mtar(i)).weights(:)'),2);
		end
		
		% gmults for the offset is given by sum of all g's, since those
		% associated with targets being fit have a +1 and those not still
		% need to be included
		gmults(:,end) = sum(g,2);

		% add "stimulus" of all 1's for offset term to stims_plus  
		offset_xtarg = length(stims_plus)+1;
		stims_plus{offset_xtarg} = ones(size(g,1),1);

		% Construct NIM for filter fitting
		nim = mnim.nim;
		nim.subunits = [mnim.Msubunits(Mtar).subunit];

		% make additional NIM-subunit to account for newly added offset
		% term
		nim.subunits(NMsubs+1) = nim.subunits(NMsubs);	% copy last subunit
		nim.subunits(end).Xtarg = offset_xtarg;			% reset Xtarg
		nim.subunits(end).filtK = 1;					% reset filter
		nim.subunits(end).NLoffset = 0;					% reset offset
		if isa(nim.subunits(end),'LRSUBUNIT')			% reset necessary params if LR subunit
			nim.subunits(end).kt = 1;
			nim.subunits(end).ksp = 1;
		end
		nim.subunits(end).NLtype = 'lin';				% reset upstream nonlinearity type
		nim.subunits(end).weight = 1;					% reset weight
		nim.subunits(end).reg_lambdas = SUBUNIT.init_reg_lambdas();	% reset reg params to all 0's

		% Add offset terms to stim_params
		stimpar1 = nim.stim_params(1);
		stimpar1.dims = [1 1 1];
		stimpar1.tent_spacing = [];
		stimpar1.up_fac = 1;  % since any up_fac is already taken into account in subunit outputs
		nim.stim_params(offset_xtarg) = stimpar1;
				
	end
	
	function [nim,stims_plus] = format4Mweightfitting( mnim, stims, Mtar, Atar )
	% Usage: [nim,gmults,stims_plus] = format4Mweightfitting( mnim, stims, Mtar, Atar )
	% 
	% setup for fitting weights on multiplicative subunits as NIM additive
	% units.
	% gmults takes care of outputs from original additive subunits and 
	% original multiplicative subunits that are not being fit. Also outputs
	% proper stim cell array.
	% Note: method is called from fit_Mweights, so no two weights acting on
	% the same additive subunit will be fit
	% INPUTS:
	%   mnim:       multNIM object whose multiplicative subunits are being fit
	%   stims:      cell array of stim matrices for original additive and
	%               multiplicative subunits
	%   Mtar:		array of indices specifying which multiplicative
	%               subunits are to be fit
	%	Atar:		cell array whose ith cell contains a subset of the 
	%				values in 'targets' field from the ith Msubunit in the 
	%				Mtar array. This subset defines which weights will be
	%				fit.
	%
	% OUTPUTS:
	%   nim:        updated NIM object that contains subunits to be fit
	%   stims_plus: stim cell array for input into the NIM.fit_filters
	%				method to determine the weights
	
		Mnontargs = setdiff(1:length(mnim.Msubunits),Mtar); % list of nontargeted mult subunits

		% Extract relevant outputs from add and mult subunits
		[~,fg_add] = mnim.get_subunit_outputs( stims );		% fg_add incorporates weight
		[~,fg_xmult,fg_mult] = mnim.calc_gmults( stims );	% fg_xmult incorporates weights (from both the NIM 'weight' property and Mweights) but not 1+ offset
															% fg_mult incorporates subunit weight but not Msubunits.weights and not 1+ offset

		% Multiply output of additive subunits and non-target mult subunits
		g = fg_add;
		for i = 1:length(Mnontargs)
			targs = mnim.Msubunits(Mnontargs(i)).targets;
			g(:,targs) = g(:,targs) .* (1 + fg_xmult{Mnontargs(i)});
		end

		% Multiply g by output of multiplicative subunits whose weights are
		% not being fit
		for i = 1:length(Mtar)
			[Anontargs,Anontarg_indx] = setdiff(mnim.Msubunits(Mtar(i)).targets,Atar{i});
			g(:,Anontargs) = g(:,Anontargs) .* (1 + fg_xmult{Mtar(i)}(:,Anontarg_indx));
		end
				
		% construct stim matrix; one stim for weights, one for offsets
		stims_plus = cell(1,2); 
		% Each weight that is being fit will be multiplied by the gmult
		% corresponding to its Atar value, as well as multiplied by the
		% (unweighted) output of its Mtar value
		Nweight_fits = length([Atar{:}]);
		weight_stim = zeros(size(g,1),Nweight_fits);
		weight_counter = 1;			% indx into weight_stim
		for i = 1:length(Mtar)
			for j = 1:length(Atar{i})
				weight_stim(:,weight_counter) = g(:,Atar{i}(j)) .* fg_mult(:,Mtar(i));
				weight_counter = weight_counter + 1;
			end
		end
		stims_plus{1} = weight_stim;
		% stims_plus for the offset is given by sum of all g's, since those
		% associated with targets being fit have a +1 and those not still
		% need to be included
		stims_plus{2} = sum(g,2);
		     
        % initialize model
		nim = mnim.nim;		% gives us appropriate spiking nonlinearity
		weight = 1;			
		NLtype = 'lin';
		NLoffset = 0;
		% weight subunit
		init_filt = randn(Nweight_fits,1)/Nweight_fits;							% this is the filt that will be fit
		Xtarg = 1;
		nim.subunits = SUBUNIT(init_filt, weight, NLtype, Xtarg, NLoffset);		% make weight subunit (resets subunits in nim object)
		% additional subunit to account for newly added offset term
		init_filt = 1;															% this filt will NOT be fit
		Xtarg = 2;
		nim.subunits(end+1) = SUBUNIT(init_filt, weight, NLtype, Xtarg, NLoffset);	% make weight subunit (resets subunits in nim object)
		
		% set stim_params field of NIM object
		stim_params = NIM.create_stim_params([1 Nweight_fits 1]);
		stim_params(2) = NIM.create_stim_params([1 1 1]);
		nim.stim_params = stim_params;
		
	end
	
end

end

