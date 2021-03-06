classdef multNIM2
% Class implementation of multiplicative NIM based on NIM class. 

properties
	nim;			% NIM or sNIM struct that contains the normal subunits
	Mnims;		% struct of multiplicative NIMs
		% nim       % actual NIM/sNIM that should multiply Mtargets
		% targets;	% array of targets in the nim struct
		% weights;	% array of weights on the output of the Mnims (one for each target)
end	

properties (Hidden)
	version = '0.4';    %source code version used to generate the model
	create_on = date;   %date model was generated
end	

% TODO


%% ******************** constructor ********************************
methods

	function mnim = multNIM2( nim, Mnims, Mtargets, Mweights )
	% Usage: mnim = multNIM2( nim, <Mnims, Mtargets>, <Mweights> )
	%
	% INPUTS:
	%   nim:        either an object of the NIM or sNIM class
	%   Mnim:       array of NIM objects that will act as multiplicative
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
			mnim.Mnims = [];
			return
		end

		% Error checking on inputs
		assert( nargin >= 3, 'Must specify targets as well as subunits' )
		
		NMsubs = length(Mnims);
		
		% Define defaults
		mnim.nim = nim;
		mnim.Mnims(1).nim = [];
		mnim.Mnims(1).targets = [];
		mnim.Mnims(1).weights = [];

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

		% Error checking on Mtargets
		assert( all(cellfun(@(x) all(ismember(x,1:length(nim.subunits))),Mtargets)), 'Invalid Mtargets.' )
		
		% Error checking/default setting on Mweights
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
		
		
		% Set multNIM properties
		mnim.nim = nim;
		mnim.Mnims = struct([]);	% start with empty struct
		for i = 1:NMsubs
			mnim.Mnims(end+1).nim = Mnims(i);
			mnim.Mnims(end).targets = Mtargets{i}(:)';  % make row vector for easy manipulation later
			mnim.Mnims(end).weights = Mweights{i}(:)';  % make row vector for easy manipulation later
		end
		
	end % method

	function mnim_out = set_MnimEI( mnim, EI, Mtar, subs )
	% Usage: mnim_out = mnim.set_MnimEI( EI, <Mtar>, <subs> )
	% Set desired subunits in Mtar as excitatory or suppressive
	% EI = 1 or -1
	
		if nargin < 3
			Mtar = 1;
		end
		Nsubs = length( mnim.Mnims(Mtar).nim.subunits );
		if nargin < 4
			subs = 1:Nsubs;
		end
		
		mnim_out = mnim;
		for nn = subs
			if ~strcmp(mnim.Mnims(Mtar).nim.subunits(nn).NLtype,'lin')
				mnim_out.Mnims(Mtar).nim.subunits(nn).weight = EI;
			end
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
			tmp = stims;  clear stims
			stims{1} = tmp;
		end

		% Append necessary options to varargin to pass to fit_filters
		varargin{end+1} = 'gain_funs';
		varargin{end+1} = mnim.calc_gmults( stims );

		% Fit filters
		mnim_out = mnim;
		mnim_out.nim = mnim.nim.fit_filters( Robs, stims, varargin{:} );    
	end
	
	function mnim2_out = fit_Mfilters( mnim2, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_Mfilters( Robs, stims, Uindx, varargin )
	%
	% Fits filters of selected MNIM. Currently only fits one mnim at a time
	% Additional varargin:
	%   'target': which Mnim to fit. Default: first one
	%   'subs': which subunits to fit within mult-NIM. Default = all
    
		% Ensure proper format of stims cell array
		if ~iscell(stims)
			tmp = stims; clear stims
			stims{1} = tmp;
		end

		if length(mnim2.Mnims) > 1
			error('This function only works with one Mnim currently.')
		end
		
		% Parse optional arguments, separating those needed in this method from those needed as inputs to fit_filters method in NIM
		[~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'target','subs'} );
		% Save indices of subunits targeted for fitting
		if isfield( parsed_inputs, 'target' ) 
			Mtar = parsed_inputs.target;
		else
			Mtar = 1;
		end
		
		assert(length(Mtar) == 1,'Can only fit single multiplicative NIM.')
		assert(ismember(Mtar,1:length(mnim2.Mnims)),'Invalid ''subs'' input')

		% Swap roles of additive and multiplicative subunits
		[nim_swap,gmults,stims_plus] = mnim2.format4Mfitting( stims, Mtar );

		NMsubs = length(mnim2.Mnims(Mtar).nim.subunits);
		if isfield( parsed_inputs, 'subs' ) 
			subs = parsed_inputs.subs;
			assert(all(ismember(subs,1:NMsubs)),'Invalid ''subs'' input')
		else
			subs = 1:NMsubs;
		end

		% Append necessary options to varargin to pass to fit_filters
		modvarargin{end+1} = 'gain_funs';
		modvarargin{end+1} = gmults;
		modvarargin{end+1} = 'subs';
		modvarargin{end+1} = subs;

		% Fit filters using NIM method
		nim_swap = nim_swap.fit_filters( Robs, stims_plus, modvarargin{:} );

		% Copy filters back to their locations
		mnim2_out = mnim2;
		mnim2_out.Mnims(Mtar).nim.subunits = nim_swap.subunits(1:NMsubs);	
		mnim2_out.nim.spkNL = nim_swap.spkNL;
		mnim2_out.nim.spk_hist = nim_swap.spk_hist;
    
		% Modify fit history
		mnim2_out.nim.fit_props.fit_type = 'Mnim';
		mnim2_out.nim.fit_history(end).fit_type = 'Mnim';    
	end

	function mnim2_out = fit_filters( mnim2, Robs, stims, varargin )
	% Use either fit_Afilters or fit_Mfilters (or fit_alt_filters)
		warning( 'Use either multNIM.fit_Afilters or fit_Mfilters/fit_Msequential (or fit_alt_filters). Defaulting to fit_Afilters.' )
		mnim2_out = mnim2.fit_Afilters( Robs, stims, varargin{:} );
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
	
	function mnim2_out = fit_Mweights( mnim2, Robs, stims, varargin )
	% Usage: mnim2_out = mnim2.fit_Mweights( Robs, stims, Uindx, varargin )
	%
	% Fits weights w_{ij} of multiplicative subunit i on additive subunit j, where the gain signal acting on 
	% additive subunit j is (1 + w_{i,j}*(output of mult subunit i)
	%
	% Note: method only handles case where at most one weight per additive subunit is fit. In the event that 
	% more than one weight is specified the method will exit with an error message.
	%
	% Optional flags:
	%	'target': Mnim to fit multiplicative weights. Default: 1
	%	'weight_targs': array that specifies which weights from the selected Mnim to fit. Should be a single  
	%     array that contains a subset of the values in 'targets' field from the (targeted) Mnim.
	%	'positive_weights': 1 to constrain weights to be positive, 0 otherwise. Default = 1.
	% 'no_normalize': 1 to prevent normalizing maximum value of weights to 1
	%
	% OUTPUTS:
	%   mnim_out:   updated multNIM object
	
		% Ensure proper format of stims cell array
		if ~iscell(stims)
			tmp = stims;  clear stims
			stims{1} = tmp;
		end

		% Parse optional arguments, separating those needed in this method from those needed as inputs to fit_filters method in NIM
		[~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'target','weight_targs','positive_weights'} );
		
		% get weight constraints
		if isfield( parsed_inputs, 'positive_weights' )
			positive_weights = parsed_inputs.positive_weights;
			assert(ismember(positive_weights,[0,1]),'''positive_weights'' flag must be set to 0 or 1')
		else
			positive_weights = 1;								% Default to constraining weights to be positive
		end
		
		% Save indices of mult subunits targeted for fitting
		if isfield( parsed_inputs, 'target' ) 
			Mtar = parsed_inputs.target;
		else
			Mtar = 1;
		end
		% Ensure values in Mtar field match possible Mnims
		assert(length(Mtar) == 1, 'Can only fit weights to 1 subunit at a time.')
		assert(all(ismember(Mtar,1:length(mnim2.Mnims))),'Invalid ''target'' input')

		% Save indices of additive subunits targeted for fitting
		if isfield( parsed_inputs, 'weight_targs' ) 
			Atar = parsed_inputs.weight_targs;
		else
			% default is all targets of each Mtar
 			Atar = mnim2.Mnims(Mtar).targets;	
		end
		
		assert(all( ismember(Atar, mnim2.Mnims(Mtar).targets)), 'Atar does not match possible targets for Msubunit %i',Mtar);
		
		% Create NIM for fitting weights; a single subunit will contain a filter with all the combined weights that are being fit
		[nim_weight,stims_plus] = format4Mweightfitting( mnim2, stims, Mtar, Atar );
		nim_weight.subunits(1).Ksign_con = positive_weights;	% set weight constraints

		% Append necessary options to varargin to pass to fit_filters
		modvarargin{end+1} = 'subs';
		modvarargin{end+1} = 1;	% just fit weights

		% Fit filters using NIM method
		nim_weight = nim_weight.fit_filters( Robs, stims_plus, modvarargin{:} );

		% Copy filters back to their locations
		mnim2_out = mnim2;
		% Mnims have not changed
		mnim2_out.nim.spkNL = nim_weight.spkNL;
		mnim2_out.nim.spk_hist = nim_weight.spk_hist;
		
		% Update Mweights, which are found in nim_weight filter
		[~,Atar_indx] = ismember(Atar,mnim2_out.Mnims(Mtar).targets); % get indices of updated weights
		mnim2_out.Mnims(Mtar).weights(Atar_indx) = nim_weight.subunits(1).filtK((1:length(Atar)))'; % filtK is column vec
    
		% Normalize weights (max-value of 1)
		if ~isfield(parsed_inputs,'no_normalize') || (parsed_inputs.no_normalize == 0)
			nrm = max(mnim2_out.Mnims(Mtar).weights(Atar_indx));
			if nrm > 0
				mnim2_out.Mnims(Mtar).weights(Atar_indx) = mnim2_out.Mnims(Mtar).weights(Atar_indx)/nrm;
			else
				warning('Weights taken to zero.')
			end
		end
			
		% Modify fit history
		mnim2_out.nim.fit_props.fit_type = 'Mweight';
		mnim2_out.nim.fit_history(end).fit_type = 'Mweight';
    
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

	function mnim_out = init_Anonpar_NLs( mnim, Xstims, varargin )
	% Usage: mnim_out = mnim.init_Anonpar_NLs( Xstims, varargin )
	
		mnim_out = mnim;
		mnim_out.nim = mnim_out.nim.init_nonpar_NLs( Xstims, varargin{:} );
	end

	function mnim_out = init_Mnonpar_NLs( mnim, Xstims, varargin )
	% Usage: mnim_out = mnim.init_Mnonpar_NLs( Xstims, varargin )
	%
	% Note that automatically sets bounds on range of nonlinearity between 0 and 1
	
		[~,parsed_options,modvarargin] = NIM.parse_varargin( varargin, {'subs'} );

		nimtmp = mnim.nim;
		NAsub = length(nimtmp.subunits);
		NMsub = length(mnim.Mnims);
		if isfield(parsed_options,'subs')
			subs = parsed_options.subs + NAsub;
		else
			subs = NAsub + (1:NMsub);
		end
		
		for nn = 1:NMsub
			nimtmp.subunits(NAsub+nn) = mnim.Mnims(nn).subunit;
		end
		
		modvarargin{end+1} = 'subs';
		modvarargin{end+1} = subs;
		
		nimtmp = nimtmp.init_nonpar_NLs( Xstims, modvarargin{:} );
		mnim_out = mnim;
		for nn = 1:NMsub
			mnim_out.Mnims(nn).subunit = nimtmp.subunits(NAsub+nn);
			% Constrain range of multiplicative subunit output
			mnim_out.Mnims(nn).subunit.NLnonpar.TBparams.NLrange = [0 1];
			TBy = mnim_out.Mnims(nn).subunit.NLnonpar.TBy;
			TBy(TBy < 0) = 0; TBy(TBy > 1) = 1;
			mnim_out.Mnims(nn).subunit.NLnonpar.TBy = TBy;
		end
	end
	
	
	function mnim_out = fit_AupstreamNLs( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_AupstreamNLs( Robs, stims, Uindx, varargin )
	
		% Ensure proper format of stims cell array
		if ~iscell(stims)
			tmp = stims;  clear stims
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
	% Note 1: Specify which Mnims to optimize using 'subs' option, numbered by their index in Mnims array
	% Note 2: method only handles case where the Mnims to optimize target unique additive subunits; if this is
	%         not the case, use fit_Msequential(which will in turn call this method properly)
    
	% Ensure proper format of stims cell array
	if ~iscell(stims)
		tmp = stims;  clear stims
		stims{1} = tmp;  
	end

	% Parse optional arguments, separating those needed in this method from those needed as inputs to fit_filters method in NIM
	[~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs'} );
	% Save indices of subunits targeted for fitting
	if isfield( parsed_inputs, 'subs' )
		Mtar = parsed_inputs.subs;
	else
		Mtar = 1:length( mnim.Mnims );
	end
	assert(all(ismember(Mtar,1:length(mnim.Mnims))),'Invalid ''subs'' input')
	NMsubs = length(Mtar);

	% Ensure multiple multiplicative units are not acting on the same additive subunit
	Atar = [mnim.Mnims(Mtar).targets];
	assert(length(Atar)==length(unique(Atar)),'Cannot simultaneously fit two mult subunits with same target')

	% Swap roles of additive and multiplicative subunits
	[nim_swap,gmults,stims_plus] = format4Mfitting( mnim, stims, Mtar );

	% Append necessary options to varargin to pass to fit_filters
	modvarargin{end+1} = 'gain_funs';
	modvarargin{end+1} = gmults;
	modvarargin{end+1} = 'subs';
	modvarargin{end+1} = 1:NMsubs;
	modvarargin{end+1} = 'rescale_nls';
	modvarargin{end+1} = 0;
	%modvarargin{pos+4} = 'fit_offsets';
	%modvarargin{pos+5} = 1;

	% fit filters using NIM method
	nim_swap = nim_swap.fit_upstreamNLs( Robs, stims_plus, modvarargin{:} );

	% Copy filters back to their locations
	mnim_out = mnim;
	for nn = 1:NMsubs
		mnim_out.Mnims(Mtar(nn)).subunit = nim_swap.subunits(nn);
	end
	%[mnim_out.Mnims(Mtar).subunit] = nim_swap.subunits(1:NMsubs);	% save mult subunits (not offset subunit)
	mnim_out.nim = nim_swap;											% save upstream/spkNL params
	mnim_out.nim.subunits = mnim.nim.subunits;							% save add subunits
    
	% modify fit history
	mnim_out.nim.fit_props.fit_type = 'Mupstream_NLs';
	mnim_out.nim.fit_history(end).fit_type = 'Mupstream_NLs';
    
	end

	function mnim_out = fit_alt_nonparNLs( mnim, Robs, stims, varargin )
	% Usage: mnim_out = fit_alt_nonparNLs( mnim, Robs, stims, varargin )
	%	
	% use flag 'mult_first' to fit Afilters before Mfilters
	% use flag 'all' to fit both filters and nonlinearities
	% Always ends with fitting multiplicative nonlinearity
		
		LLtol = 0.0002; MAXiter = 12;

		% Check to see if silent (for alt function)
		[~,parsed_options,modvarargin] = NIM.parse_varargin( varargin, {'silent','mult_first','all'} );
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
		
		fitall = isfield( parsed_options, 'all' );
		mnim_out = mnim;
		if isfield( parsed_options, 'mult_first')
			mnim_out = mnim_out.fit_AupstreamNLs( Robs, stims, modvarargin{:} );
			if fitall
				mnim_out = mnim_out.fit_alt_filters( Robs, stims, modvarargin{:} );
			end
		end
		
		iter = 1;
		while (((LL-LLpast) > LLtol) && (iter < MAXiter))

			mnim_out = mnim_out.fit_AupstreamNLs( Robs, stims, modvarargin{:} );
			if fitall
				mnim_out = mnim_out.fit_alt_filters( Robs, stims, modvarargin{:} );
			end
			
			mnim_out = mnim_out.fit_MupstreamNLs( Robs, stims, modvarargin{:} );
			if fitall
				mnim_out = mnim_out.fit_alt_filters( Robs, stims, modvarargin{:} );
			end
			
			LLpast = LL;
			LL = mnim_out.nim.fit_props.LL;
			iter = iter + 1;

			if ~silent
				fprintf( '  Iter %2d: LL = %f\n', iter, LL )
			end	
		end
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
		mnim_out.nim = mnim.nim.reg_path2( Robs, stims, Uindx, XVindx, varargin{:} );
	end
		
	function mnim2_out = reg_pathM( mnim2, Robs, stims, Uindx, XVindx, varargin )
	% Usage: mnim2 = reg_pathM( mnim2, Robs, stims, Uindx, XVindx, varargin )
	%
	% Parse optional arguments, separating those needed in this method from those needed as inputs 
	% to fit_filters method in NIM
	% Specify single "target" for the Mnim to fit, as potentially 'subs' for the separate subunits within
	
		[~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'target'} );
		% Save indices of subunits targeted for fitting
		if isfield( parsed_inputs, 'target' )
			Mtar = parsed_inputs.target;
		else
			Mtar = 1; %1:length( mnim2.Mnims );
		end
		assert(length(Mtar) == 1,'Can only use one Mtarget here.')
		assert(all(ismember(Mtar,1:length(mnim2.Mnims))),'Invalid ''target'' input')
		%NMnims = length(Mtar);
		
    % swap roles of additive and multiplicative subunits
    [nim_swap,gmults,stims_plus] = format4Mfitting( mnim2, stims, Mtar );

    % append necessary options to varargin to pass to fit_filters
    modvarargin{end+1} = 'gain_funs';
    modvarargin{end+1} = gmults;
    modvarargin{end+1} = 'subs';
    modvarargin{end+1} = 1:length(mnim2.Mnims(Mtar).nim.subunits);

    % Use NIM method reg_path
    nim_swap = nim_swap.reg_path2( Robs, stims_plus, Uindx, XVindx, modvarargin{:} );

		% Copy filters back to their locations
		mnim2_out = mnim2;
		mnim2_out.Mnims(Mtar).nim.subunits = nim_swap.subunits(1:length(mnim2.Mnims(Mtar).nim.subunits));	
		mnim2_out.nim.spkNL = nim_swap.spkNL;
		mnim2_out.nim.spk_hist = nim_swap.spk_hist;
		    
    % Modify fit history
    mnim2_out.nim.fit_props.fit_type = 'Mfilt_RegPath';
    mnim2_out.nim.fit_history(end).fit_type = 'Mfilt_RegPath';
	end
	    
	function mnim_out = fit_Msequential( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_Msequential( Robs, stims, Uindx, varargin )
	% Fits filters or upstream nonlinearities of multiplicative subunit, and handles case 
	% where the Mnims to optimize target the same additive subunits
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
	%               Mnims acting on the same additive subunit, the
	%               Msubunit with the smallest index will by default be fit first.
	%           option 2 - Nx1 cell array of Msubunit indices indicating 
	%               which Mnims should be fit during N calls to
	%               fit_Mfilters. Indices in each cell of the array will be
	%               checked for uniqueness of targets; if nonuniqueness is
	%               found, that cell is divided into one or more cells and
	%               the rest of the fitting order will be unaffected.
	%           option 3 - if subs is empty or 'subs' is not listed as an
	%               optional flag, all Mnims are selected and the
	%               method defaults to the behavior in option 1
	%       ('component','component'): string 'filt' or 'upstreamNLs'
	%               specifying which component of the Mnims to fit;
	%               defaults to 'filt'
	% 
	% OUTPUTS:
	%   mnim_out:   updated multNIM object
    
		% Ensure proper format of stims cell array
		if ~iscell(stims)
			tmp = stims;  clear stims
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
				Mtar = {1:length(mnim.Mnims)};					% use all subunits if empty argument
			elseif ~iscell(Mtar)
				Mtar = mat2cell(Mtar,size(Mtar,1),size(Mtar,2));	% if single vector has been input, convert to cell array
			end
		else
			Mtar = {1:length(mnim.Mnims)};						% use all subunits if flag not set
		end        
		Nfits = length(Mtar);	% total number of fits; Mtar is cell array, so Nfits might equal 1 even if fitting multiple Mnims

		% Ensure multiple multiplicative units are not acting on the same additive subunit; if they do, split Mtar
		reset_order_flag = 0;
		i = 1; % counter for number of fits
		while ~reset_order_flag && i <= Nfits
			% get additive subunits acted on by current set of subs indices
			Atar = [mnim.Mnims(Mtar{i}).targets];
			if length(Atar)~=length(unique(Atar))
				%warning('Cannot simultaneously fit two mult subunits with same target; changing fit order')
				reset_order_flag = 1;
				break
			end
			i = i+1; % update counter
		end

		% Reset order of Mtar if necessary, with lower indices of Mnims taking priority
		if reset_order_flag
			remaining_Mtargets = unique([Mtar{:}]); % get sorted list of all targeted Mnims
			Mtar = {};  % reset Mtar
			% Loop through targets until none are left
			while ~isempty(remaining_Mtargets)
				curr_Atargs = [];
				rm_indxs = [];
				for i = 1:length(remaining_Mtargets)
					i_Atargs = mnim.Mnims(remaining_Mtargets(i)).targets;
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
    
	function mnim_out = fit_Msequential_weights( mnim, Robs, stims, varargin )
	% Usage: mnim_out = mnim.fit_Msequential_weights( Robs, stims, Uindx, varargin )
	% Fits weights of multiplicative subunits, and handles case where the 
	% desired weights to optimize target the same additive subunits.
	%
	% INPUTS:
	%   Robs:
	%   stim:
	%   optional flags:
	%       any optional flag argument used for fit_Mweights method
	%       'subs':  
	%           option 1 - array of Msubunit indices indicating which
	%               subunits' weights will be optimized. If all selected 
	%				subunits act on unique additive subunits, fit_Mweights
	%				is called. If the Mnims selected do NOT act on 
	%				unique additive subunits, there are two options:
	%				1) select which weights will be optimized using the 
	%				'weight_targs' flag (see below)
	%				2) if 'weight_targs' flag is not set or is not feasible,
	%				the indices are arranged so that all weights of 
	%				specified Mnims	will be fit once by calling 
	%				fit_Mweights sequentially; in the case of multiple 
	%				Mnims acting on a single additive subunit, the 
	%				order in which the weights are fit follow the
	%				order of the Mnims in the input
	%           option 2 - Nx1 cell array of Msubunit indices indicating 
	%               which Mnims should be fit during each of N calls to
	%               fit_Mweights. In this case the steps outlined in option
	%               1 above are carried out for each cell.
	%           option 3 - if subs is empty or 'subs' is not listed as an
	%               optional flag, all Mnims are selected and the
	%               method defaults to the behavior in option 1
	%		'weight_targs':
	%			option 1 - matches 'subs' option 1; a 1xN array of Msubunit
	%				indices is specified, and 'weight_targs' should be a 1xN
	%				cell array where the indices in each cell specify which
	%				weights of the corresponding Msubunit should be fit.
	%			option 2 - matches 'subs' option 2; a 1xN cell array. Each
	%				cell is itself a cell array, such that the indices in 
	%				cell{n}{m} are the weights from Msubunit m that should
	%				be fit on the nth call to fit_Mweights. See below for
	%				an example.
	%			option 3 - 'weight_targs' is empty or not set as an
	%				optional flag; in this case, it is assumed that all
	%				weights for specified Mnims are to be fit.
	%	'positive_weights': 1 to constrain weights to be positive, 0
	%			otherwise. Default = 1.
	%
	%
	%	Example: Mnims(1).targets = [1 2 3];
	%			 Mnims(2).targets = [1 3];
	%			 Mnims(3).targets = [4];
	%
	%   Using option1/option1
	%	mnim.fit_Msequential_weights(Robs,stim,'subs',[1 2 3],'weight_targs',{1,3,4})
	%		will fit Msubunit(1) weight on 1st additive subunit,
	%		Msubunit(2) weight on 3rd additive subunit and Msubunit(3) 
	%		weight on 4th additive subunit. All weights will be fit 
	%		simultaneously.
	%	mnim.fit_Msequential_weights(Robs,stim,'subs',[1 2 3],'weight_targs',{1,[1,3],4})
	%		In this case not all specified weights can be fit the same time
	%		since the weight from Msubunit(1) to additive subunit 1 and the
	%		weight from Msubunit(2) to additive subunit 1 are both specified.
	%		Instead, the fitting order will be rearranged as explained 
	%		above, so that the equivalent call would be
	%		mnim.fit_Msequential_weights(...,'subs',{[1 2 3],[2]},'weight_targs',{{1,3,4},{1})
	%
	%	Using option2/option2
	%	mnim.fit_Msequential_weights(...,'subs',{[1 3],2,1},'weight_targs',{{[1 2 3],4},{[1 3]},{[1 3]}})
	%		will call fit_Mweights 3 times ('subs' is a 1x3 cell array). 
	%		On the first call, weights from Msubunit(1) on additive
	%			subunits 1,2 and 3 will be fit, along with weights from
	%			Msubunit(3) on additive subunit 4.
	%		On the second call, weights from Msubunit(2) on additive
	%			subunits 1 and 3 will be fit
	%		On the third call, weights from Msubunit(1) on additive
	%			subunits 1 and 3 will be fit
	%
	% OUTPUTS:
	%   mnim_out:   updated multNIM object
    
		% Ensure proper format of stims cell array
		if ~iscell(stims)
			tmp = stims; clear stims
			stims{1} = tmp;
		end

		% Parse optional arguments, separating those needed in this method from those needed as inputs to fit_filters method in NIM
		[~,parsed_inputs,modvarargin] = NIM.parse_varargin( varargin, {'subs','weight_targs','weight_targs'} );

		% get weight constraints
		if isfield( parsed_inputs, 'positive_weights' )
			positive_weights = parsed_inputs.positive_weights;
			assert(ismember(positive_weights,[0,1]),'''positive_weights'' flag must be set to 0 or 1')
		else
			positive_weights = 1;								% Default to constraining weights to be positive
		end
		
		% Save indices of mult subunits targeted for fitting
		if isfield( parsed_inputs, 'subs' ) 
			Mtar = parsed_inputs.subs;
			% turn into cell array if not already
			if ~iscell(Mtar)
				tmp = Mtar;
				clear Mtar
				Mtar{1} = tmp;									% make Mtar a 1x1 cell array
			end
		else
			Mtar{1} = 1:length( mnim.Mnims );				% Default to all Mnims
		end
		% Ensure values in Mtar cells match possible Mnims
		for i = 1:length(Mtar)
			assert(all(ismember(Mtar{i},1:length(mnim.Mnims))),'Invalid ''subs'' input')
		end
		
		% Save indices of additive subunits targeted for fitting
		if isfield( parsed_inputs, 'weight_targs' ) 
			Atar = parsed_inputs.weight_targs;
			% turn into cell array if not already
			if isempty(Atar)
				% default is all targets of each Msubunit in each Mtar cell
				for i = 1:length(Mtar)
					[Atar{i}{1:length(Mtar{i})}] = mnim.Mnims(Mtar{i}).targets; % each cell of Atar{i} will contain targets from one Msubunit
				end
			elseif ~iscell(Atar)
				% assume 1-1 correspondence between values in this array and the Mnims in Mtar{1}
				tmp = Atar;  clear Atar
				Atar{1}{1} = tmp;								% make Atar{1} a 1x1 cell array
			elseif ~iscell(Atar{1})
				% assume 1-1 correspondence between values in this cell array and the Mnims in Mtar{1}
				tmp = Atar;  clear Atar
				Atar{1} = tmp;
			end
			% check for proper dimensions; should only be one Atar cell per Mtar cell
			assert(length(Mtar)==length(Atar),'Mismatch between ''subs'' and ''weight_targs'' input dimensions')
		else
			% default is all targets of each Msubunit in each Mtar cell
			for i = 1:length(Mtar)
				[Atar{i}{1:length(Mtar{i})}] = mnim.Mnims(Mtar{i}).targets; % each cell of Atar{i} will contain targets from one Msubunit
			end
		end
		
		% Ensure specified Atars match up with possible Msubunit targets
		for i = 1:length(Mtar)
			for j = 1:length(Mtar{i})
				assert(all( ismember(Atar{i}{j}, mnim.Mnims(Mtar{i}(j)).targets)), 'Atar index does not match possible targets for Msubunit %i\n',Mtar{i}(j));
			end
		end
			
		% Loop through Mtar, breaking up the calls to fit_Mweights if necessary
		for i = 1:length(Mtar)
			
			% Check to see if multiple weights are being fit on a single additive subunit; if so, flag and take care of below
			reset_order_flag = 0;		
			add_targets = [];
			for j = 1:length(Mtar{i})
				add_targets = [add_targets Atar{i}{j}];
				if length(add_targets) ~= length(unique(add_targets))
					% at least one subunit has been targeted more than once
					%warning('Cannot simultaneously fit two weights acting on same additive subunit; changing fit order\n')
					reset_order_flag = 1;
					break
				end
			end
			
			% Partition Mtar{i} if necessary, with lower indices of Mnims taking priority
			temp_Mtar = {};					% store new Mtar; need a cell array since there will potentially be multiple fits
			temp_Atar = {};					% store new Atar
			if reset_order_flag
				remaining_Mtargs = Mtar{i};	% get list of all targeted Mnims
				remaining_Atargs = Atar{i};	% get corresponding list (still a cell array) of additive targets
				% Loop through Mtar until no weights are left to fit
				while ~isempty(remaining_Mtargs)
					curr_Mtargs = [];		% current set of targeted Mnims
					curr_Atargs = {};		% corresponding set of targets of those targeted subunits
					rm_indxs = [];			% indices of remaining_Mtargs/remaining_Atargs to remove
					for j = 1:length(remaining_Mtargs)
						[~,Atargs_indx] = ismember(remaining_Atargs{j},[curr_Atargs{:}]);
						if any(Atargs_indx == 0)
							% if any members of remaining_Atargs{j} are NOT in the set of curr_Atargs, put them there
							curr_Mtargs = [curr_Mtargs remaining_Mtargs(j)];	% add this Msubunit to temp_Mtar
							curr_Atargs{end+1} = remaining_Atargs{j}(Atargs_indx == 0);
							remaining_Atargs{j}(Atargs_indx == 0) = [];
							if isempty(remaining_Atargs{j})
								% clear remaining A- and M- targs if nothing left
								rm_indxs = [rm_indxs j];
							end
						end
					end
					remaining_Atargs(rm_indxs) = [];	% remove cells of remaining_Atargs that have been fully assigned
					remaining_Mtargs(rm_indxs) = [];	% remove corresponding remaining_Mtargs
					temp_Atar{end+1} = curr_Atargs;		% save curr_Atargs
					temp_Mtar{end+1} = curr_Mtargs;		% save curr_Mtargs
				end
			else
				% we do not need to reset the order; change variable names for compatibility with reset_order variables above
				temp_Mtar{1} = Mtar{i};
				temp_Atar{1}{1} = Atar{i};
			end
					
			% set number of calls to fit_Mweights
			Nfits = length(temp_Mtar); 
			
			modvarargin{end+1} = 'subs';			% 'subs' option was removed by parse_varargin
			counter = length(modvarargin);
			modvarargin{end+1} = [];
			modvarargin{end+1} = 'weight_targs';    % 'weight_targs' option was removed by parse_varargin
			modvarargin{end+1} = [];
			for n = 1:Nfits
				modvarargin{counter+1} = temp_Mtar{n};
				modvarargin{counter+3} = temp_Atar{n};
				mnim = mnim.fit_Mweights(Robs,stims,modvarargin{:});
				% modify fit history
				mnim.nim.fit_props.fit_type = 'Mweight';
				mnim.nim.fit_history(end).fit_type = 'Mweight';
			end
		end
			
		mnim_out = mnim;

	end
	
end

%% ********************  Helper Methods ********************************
methods
    
	function mnim2 = add_mnim( mnim2, nim, targets, weights, Xtargs )
	% Usage: mnim = mnim2.add_mnim( nim, <targets>, <weights>, <Xtargs>)
	%
	% INPUTS:
	%   nim:        the NIM that will be added to be multiply the targets
	%               of an existing subunit to clone for a multiplicative subunit 
	%   targets:    array specifying additive subunits that new Msubunit multiplies; defaults 
	%               to all subunits
	%	  weights:    array specifying weights on subunit output before 1 is added; defaults to 
	%               a weight vector of ones
	%   Xtargs:      Xtargets (refering to Xstim) for all subunits in the added Mnim. Default is
	%               to leave as-is. If length 1, all Xtargs will be the same
	% OUTPUTS:
	%   mnim:       updated multNIM object with new Msubunit

		if (nargin < 3) || isempty(targets)
			targets = 1:length(mnim2.nim.subunits);
		end
		if (nargin < 4) || isempty(weights)
			weights = ones(1,length(targets));
		end
		if (nargin < 5) 
			Xtargs = [];
		end
		% Error checking on inputs
		assert( all(targets <= length(mnim2.nim.subunits)), 'Mtargets must be existing subunits.' )
		
		mnim2.Mnims(end+1).nim = nim;
		mnim2.Mnims(end).nim.stim_params = mnim2.nim.stim_params;
		mnim2.Mnims(end).targets = targets;
		mnim2.Mnims(end).weights = weights;
		if ~isempty(Xtargs)
			Nsubs = length(nim.subunits);
			if length(Xtargs) == 1
				Xtargs = ones(Nsubs,1)*Xtargs;
			else
				assert(length(Xtargs)==Nsubs,'Need one Xtarg or one Xtarg corresponding to each subunit.' )
			end
			for nn = 1:length(Xtargs)
				mnim2.Mnims(end).nim.subunits(nn).Xtarg = Xtargs(nn);
			end
		end
		
	end
    
	function Msub = clone_to_Msubunit( mnim, subN, weight, filter_flip )
	% Usage: Msub = mnim.clone_to_Msubunit( subN, <weight>, <filter_flip> ) 
	% Uses an existing subunit of an NIM object as an Msubunit
	%
	% INPUTS:
	%   subN:         index of NIM subunit to clone into an Msubunit
	%   <weight>:	    desired weight (+/-1) of new subunit; this is
	%					different from Mnims(_).weights, which weight the
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

	function [LL, pred_rate, mod_internals, LL_data] = eval_model( mnim2, Robs, stims, varargin )
	%	Usage: [LL, pred_rate, mod_internals, LL_data] = mnim.eval_model( Robs, stims, varargin )
	%
	% OUTPUT:
	%   LL:
	%   pred_rate:
	%   mod_internals: struct with following information gint ...

		if ~iscell(stims)
			tmp = stims;  clear stims
			stims{1} = tmp;
		end

		modvarargin = varargin;
		gmults = mnim2.calc_gmults( stims );
		modvarargin{end+1} = 'gain_funs';
		modvarargin{end+1} = gmults;

		[LL,pred_rate,mod_internals,LL_data] = mnim2.nim.eval_model( Robs, stims, modvarargin{:} );
		mod_internals.gain_funs = gmults;
		
		% Multiplicative NIM output information
		if nargout > 2
			NMnims = length(mnim2.Mnims);
			mod_internals.M_outs = zeros(length(pred_rate),NMnims);
			for nn = 1:NMnims
				[~,~,Mmodints] = mnim2.Mnims(nn).nim.eval_model( [], stims, varargin{:} );
				mod_internals.M_outs(:,nn) = Mmodints.G;
				mod_internals.M_gint = Mmodints.gint; % Note that these are not weighted by Mweights!
				mod_internals.M_fgint = Mmodints.fgint; % Note that these are not weighted by Mweights!
			end
		end
	end
    
	function [sub_outs,fg_add,gmults] = get_subunit_outputs( mnim, stims )
	% Usage: [sub_outs,fg_add,gmults] = mnim.get_subunit_outputs( stims )
	% 
	% Calculates output of all subunits (additive and multiplicative) as well as separate additive and 
	% multiplicative components; called by format4Mfitting
	%
	% INPUTS:
	%   stims:      cell array of stim matrices for mnim model
	% 
	% OUTPUTS:
	%   sub_outs:   T x num_add_subunits matrix of additive subunit outputs
	%					multiplied by their corresponding gain functions
	%   fg_add:     T x num_add_subunits matrix of additive subunit outputs
	%					(includes multiplication by subunit weight)
	%   gmults:     T x num_add_subunits matrix of gmults from Msubunit outputs

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
		for nn = 1:length(mnim.Mnims)
			sub_outs(:,mnim.Mnims(nn).targets) = fg_add(:,mnim.Mnims(nn).targets) .* gmults(:,mnim.Mnims(nn).targets); 
		end
	end

	function nrmstruct = subunit_filter_norms( mnim )
	% Usage: nrms = subunit_filter_norms( mnim )
	% Returns struct containing arrays of subunit filter magnitudes: Anrms and Mnrms, 
	% and AMmatrix, which looks at coupling

		NAsubs = length(mnim.nim.subunits);
		NMsubs = length(mnim.Mnims);
		nrmstruct.Anrms = mnim.nim.subunit_filter_norms();
		nrmstruct.Mnrms = zeros(1,NMsubs);
		nrmstruct.AMmatrix = zeros(NMsubs,NAsubs);
		for nn = 1:NMsubs
			nrmstruct.Mnrms(nn) = mnim.Mnims(nn).subunit.filter_norm();
			nrmstruct.AMmatrix(nn,mnim.Mnims(nn).targets) = nrmstruct.Mnrms(nn)*mnim.Mnims(nn).weights;
		end
	end

	function mnim = init_nonpar_NLs( mnim, stims, varargin )
	% Usage: mnim = mnim.init_nonpar_NLs( stims, varargin )
	% Initializes the specified model subunits to have nonparametric (tent-basis) upstream NLs,
	% inherited from NIM version. 
	%
	% Note: default is initializes all subunits and Mnims. Change this through optional flags
	%
	% INPUTS: 
	%   stims: cell array of stimuli
	%   optional flags:
	%       ('subs',sub_inds): Index values of set of subunits to make nonpar (default is all)
	%       ('Msubs',sub_inds): Index values of set of Mnims to make nonpar (default is all)
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
			Msubs = 1:length(mnim.Mnims);
		end
		if ~isempty(Msubs)
			[~,~,modvarargin] = NIM.parse_varargin( modvarargin, {'subs'} );
			modvarargin{end+1} = 'subs';
			modvarargin{end+1} = Msubs;
			nimtmp = mnim.nim;
			nimtmp.subunits = [mnim.Mnims(:).subunit];
			nimtmp = nimtmp.init_nonpar_NLs( stims, modvarargin{:} );
			[mnim.Mnims(:).subunit] = nimtmp.subunits;
		end 
	end
		
	function display_model( mnim2, stims, Robs )
	% Usage: [] = mnim2.display_model( Xstim, Robs )
	% Model display function that plots additive subunits in left column and multiplicative in right column
	% Tis a simple display, and could be elaborated...
	
		if nargin < 3, Robs = [];	end
		if (nargin < 2) || isempty(stims)
			mod_outs = [];
		else
			[~,~,mod_outs] = mnim2.eval_model( Robs, stims );
		end
		
		extra_plots = [(mnim2.nim.spk_hist.spkhstlen > 0) ~isempty(mod_outs)]; % will be spike-history or spkNL plot?

		NMmods = length(mnim2.Mnims);
		NAsubs = length(mnim2.nim.subunits); NMsubs = 0;
		for nn = 1:NMmods;
			NMsubs = max([NMsubs length(mnim2.Mnims(nn).nim.subunits)]);
		end		
		Nrows = max([NAsubs NMsubs]);
		Ncols = 3*(1+NMmods);
		if sum(extra_plots) > 0
			Nrows = max([Nrows sum(extra_plots)]);
			Ncols = Ncols + 1;
		end

		figure
		% Plot Additive subunits
		for nn = 1:NAsubs
			dims = mnim2.nim.stim_params(mnim2.nim.subunits(nn).Xtarg).dims;
			mnim2.nim.subunits(nn).display_filter( dims, [Nrows Ncols (nn-1)*Ncols+1], 'notitle', 1 );
			subplot( Nrows, Ncols, (nn-1)*Ncols+3 )
			if isempty(mod_outs)
				mnim2.nim.subunits(nn).display_NL();
			else
				mnim2.nim.subunits(nn).display_NL( mod_outs.gint(:,nn) );
			end
			%subplot( Nrows, Ncols, (nn-1)*Ncols+1 )
			if mnim2.nim.subunits(nn).weight > 0
				title( sprintf( 'Exc #%d', nn ) )
			else
				title( sprintf( 'Sup #%d', nn ) )
			end
		end

		% Plot Mult nims
		for nn = 1:NMmods
			for mm = 1:length(mnim2.Mnims(nn).nim.subunits)
				dims = mnim2.Mnims(nn).nim.stim_params(mnim2.Mnims(nn).nim.subunits(mm).Xtarg).dims;
				mnim2.Mnims(nn).nim.subunits(mm).display_filter( dims, [Nrows Ncols (mm-1)*Ncols+3*nn+1], 'notitle', 1 );
				subplot( Nrows, Ncols, (mm-1)*Ncols+3*nn+3 )
				if isempty(mod_outs)
					mnim2.Mnims(nn).nim.subunits(mm).display_NL( 'y_offset',1.0,'sign',1 );
				else
					mnim2.Mnims(nn).nim.subunits(mm).display_NL( mod_outs.M_gint{nn}(:,nn), 'y_offset',1.0,'sign',1 );
				end
			end
			subplot( Nrows, Ncols, 3*nn+1 )
			title( sprintf( 'Mult #%d -> %d', nn, mnim2.Mnims(nn).targets(1) ) )  % at the moment only displays first Mtarget
		end

		% Plot spkNL
		if sum(extra_plots) == 0
			return
		end
		subplot( Nrows, Ncols, Ncols );
		if extra_plots(2) > 0
			mnim2.nim.display_spkNL( mod_outs.G );
			title( 'Spk NL' )
			if extra_plots(1) > 0
				subplot( Nrows, Ncols, 2*Ncols );
			end
		end
		if extra_plots(1) > 0
			mnim2.nim.display_spike_history();
		end
	end

end

%% ********************  hidden methods ********************************
methods (Hidden)
	
	function [gmults, fg_xmult, fg_mult] = calc_gmults( mnim2, stims )
	% Usage: gmults = mnim.calc_gmults( stims )
	% Hidden method used to calculate gains that will then be passed to an NIM fitting method or eval method
	% 
	% INPUTS:
	%   stims:    cell array of stim matrices for model fitting
	% OUTPUTS:
	%   gmults:   T x num_add_subunits matrix of gmults, one for each additive subunit in mnim.nim 
	%				      NIM object used by get_subunit_output and format4Mfitting
	%   fg_xmult: "Extended" fg_mult; 1 x num_Mnims cell array of multiplicative subunit outputs.
	%				      Cell i contains the output of Msubunit(i) for ALL its targets (a T x num_targs 
	%             matrix), since these outputs may be different depending on the weights (col i 
	%				      is w_i*f() ) used by format4Mfitting
	%   fg_mult:  T x num_Mnims matrix of Msubunit outputs
	%				      (includes multiplication by subunit weight) used by format4Mweightfitting
	
		% Set default gmults
		[~,~,mod_int_add] = mnim2.nim.eval_model( [], stims );	% cheap way to get size of gmults for diff nim classes
		gmults = ones(size(mod_int_add.gint));  % default gains of 1 for every subunit 
		fg_mult  = zeros(size(gmults,1),length(mnim2.Mnims));
		fg_xmult = cell(1,length(mnim2.Mnims));

		for nn = 1:length(mnim2.Mnims)
			targets = mnim2.Mnims(nn).targets;
			weights = mnim2.Mnims(nn).weights;
			[~,~,mod_internals] = mnim2.Mnims(nn).nim.eval_model( [], stims );
			fg_mult(:,nn) = mod_internals.G;
			fg_xmult{nn} = bsxfun(@times,repmat(fg_mult(:,nn),1,length(targets)),weights(:)');
			gmults(:,targets) = gmults(:,targets) .* (1+fg_xmult{nn});
		end
		
	end % method
	
	function [nim_tmp,gmults,stims_plus] = format4Mfitting( mnim2, stims, Mtar )
	% Usage: [nim,gmults,stims_plus] = mnim.format4Mfitting( stims, Mtar )
	% 
	% setup for fitting multiplicative subunits as NIM additive units with gmults taking care 
	% of outputs from original additive subunits and original multiplicative subunits that are 
	% not being fit. Also outputs proper stim cell array.
	% Note: method is called from fit_Mfilters, so no two multiplicative subunits specified by 
	%   Mtargets should have the same additive subunit as a target (though a given Mtarget can 
	%   have multiple additive subunits as targets)
	%
	% INPUTS:
	%   mnim:   multNIM object whose multiplicative subunits are being fit
	%   stims:  cell array of stim matrices for original additive and multiplicative subunits
	%   Mtar:		array of indices specifying which multiplicative subunits are to be fit
	%
	% OUTPUTS:
	%   nim:        updated NIM object that contains subunits to be fit
	%   gmults:     T x num_Mtargs matrix of gmult values obtained from
	%               additive subunits and nontarget multiplicative subunits
	%   stims_plus: same cell array as stims with an additional stim matrix
	%               to properly take into account offsets due to shuffling around model terms
	
		Mnontargs = setdiff(1:length(mnim2.Mnims),Mtar); % list of nontargeted mult subunits

		% Ensure proper format of stims cell array
		if ~iscell(stims)
			stims_plus{1} = stims;
		else
			stims_plus = stims;
		end

		% Extract relevant outputs from add and mult subunits
		[~,fg_add] = mnim2.get_subunit_outputs( stims ); % fg_add incorporates weight
		[~,fg_xmult] = mnim2.calc_gmults( stims );		% fg_xmult incorporates weights (from both the NIM 'weight' property and Mweights) but not 1+ offset

		% Multiply output of additive subunits and non-target mult subunits
		g = fg_add;
		for i = Mnontargs
			targs = mnim2.Mnims(i).targets;
			g(:,targs) = g(:,targs).*(1+fg_xmult{i});
		end

		% Now each Mtarg should be multiplied by the weighted sum of the g's corresponding to its targets
		NMsubs = length(mnim2.Mnims(Mtar).nim.subunits);
		gmults = ones(size(g,1),NMsubs+1);  % extra-dim of ones is to multiply additive offset term of all 1's
		for nn = 1:NMsubs
			%gmults(:,i) = sum(bsxfun(@times,g(:,mnim2.Mnims(Mtar).targets),mnim2.Mnims(Mtar).weights(:)'),2);
			gmults(:,nn) = sum(bsxfun(@times,g(:,mnim2.Mnims(Mtar).targets),mnim2.Mnims(Mtar).weights(:)'),2);
		end
		
		% gmults for the offset is given by sum of all g's, since those associated with targets being fit 
		% have a +1 and those not still need to be included
		gmults(:,end) = sum(g,2);

		% add "stimulus" of all 1's for offset term to stims_plus (because gmult will take care of it)
		offset_xtarg = length(stims_plus)+1;
		stims_plus{offset_xtarg} = ones(size(g,1),1);

		% Construct NIM for filter fitting
		nim_tmp = mnim2.nim;
		%nim.subunits = [mnim2.Mnims(Mtar).subunit];
		nim_tmp.subunits = mnim2.Mnims(Mtar).nim.subunits;

		% Make additional NIM-subunit to account for newly added offset term
		nim_tmp.subunits(NMsubs+1) = nim_tmp.subunits(NMsubs);	% copy last subunit
		nim_tmp.subunits(end).Xtarg = offset_xtarg;			% reset Xtarg
		nim_tmp.subunits(end).filtK = 1;					% reset filter
		nim_tmp.subunits(end).NLoffset = 0;					% reset offset
		if isa(nim_tmp.subunits(end),'LRSUBUNIT')			% reset necessary params if LR subunit
			nim_tmp.subunits(end).kt = 1;
			nim_tmp.subunits(end).ksp = 1;
		end
		nim_tmp.subunits(end).NLtype = 'lin';				% reset upstream nonlinearity type
		nim_tmp.subunits(end).weight = 1;					% reset weight
		nim_tmp.subunits(end).reg_lambdas = SUBUNIT.init_reg_lambdas();	% reset reg params to all 0's

		% Add offset terms to stim_params
		stimpar1 = nim_tmp.stim_params(1);
		stimpar1.dims = [1 1 1];
		stimpar1.tent_spacing = [];
		stimpar1.up_fac = 1;  % since any up_fac is already taken into account in subunit outputs
		nim_tmp.stim_params(offset_xtarg) = stimpar1;
	end
	
	function [nim,stims_plus] = format4Mweightfitting( mnim2, stims, Mtar, Atar )
	% Usage: [nim,stims_plus] = mnim2.format4Mweightfitting( stims, Mtar, Atar )
	% 
	% setup for fitting weights on multiplicative subunits as NIM additive units.
	% not being fit. Also outputs proper stim cell array.
	%
	% Note: method is called from fit_Mweights, so no two weights acting on the same additive subunit will be fit
	% 
	% INPUTS:
	%   mnim:  multNIM object whose multiplicative subunits are being fit
	%   stims: cell array of stim matrices for original additive and multiplicative subunits
	%   Mtar:  index specifying which multiplicative NIM weights are to be fit
	%	  Atar:  array whose that contains a subset of the values in 'targets' field from the targeted MNIM
	%				   This subset defines which weights will be fit.
	%
	% OUTPUTS:
	%   nim:        updated NIM object that contains subunits to be fit
	%   stims_plus: stim cell array for input into the NIM.fit_filters method to determine the weights	
	
		Mnontargs = setdiff(1:length(mnim2.Mnims),Mtar); % list of nontargeted mult subunits

		% Extract relevant outputs from add and mult subunits
		[~,fg_add] = mnim2.get_subunit_outputs( stims );		% fg_add incorporates weight
		[~,fg_xmult,fg_mult] = mnim2.calc_gmults( stims );	% fg_xmult incorporates weights (from both the NIM 'weight' property and Mweights) but not 1+ offset
		% fg_mult incorporates subunit weight but not Mnims.weights and not 1+ offset

		% Multiply output of additive subunits and non-target mult subunits
		g = fg_add;
		for ii = 1:length(Mnontargs)
			targs = mnim2.Mnims(Mnontargs(ii)).targets;
			g(:,targs) = g(:,targs) .* (1 + fg_xmult{Mnontargs(ii)});
		end

		% Multiply g by output of multiplicative subunits whose weights are not being fit
		for ii = 1:length(Mtar)
			[Anontargs,Anontarg_indx] = setdiff(mnim2.Mnims(Mtar(ii)).targets,Atar);
			g(:,Anontargs) = g(:,Anontargs) .* (1 + fg_xmult{Mtar(ii)}(:,Anontarg_indx));
		end

		% construct stim matrix; one stim for weights, one for offsets
		stims_plus = cell(1,2); 
		% Each weight that is being fit will be multiplied by the gmult corresponding to its Atar value, as well as multiplied by the
		% (unweighted) output of its Mtar value
		Nweight_fits = length(Atar);
		weight_stim = zeros(size(g,1),Nweight_fits);
		weight_counter = 1;			% indx into weight_stim
		for jj = 1:length(Atar)
			weight_stim(:,weight_counter) = g(:,Atar(jj)) .* fg_mult(:,Mtar);
			weight_counter = weight_counter + 1;
		end
		stims_plus{1} = weight_stim;
		% stims_plus for the offset is given by sum of all g's, since those associated with targets being fit have a +1 and those
		% not still need to be included
		stims_plus{2} = sum(g,2);
  
		% initialize model
		nim = mnim2.nim;		% gives us appropriate spiking nonlinearity
		weight = 1;			
		NLtype = 'lin';
		NLoffset = 0;
		% weight subunit
		init_filt = randn(Nweight_fits,1)/Nweight_fits;							% this is the filt that will be fit
		Xtarg = 1;
		nim.subunits = SUBUNIT(init_filt, weight, NLtype, Xtarg, NLoffset);		% make weight subunit (resets subunits in nim object)
		% additional subunit to account for newly added offset term
		init_filt = 1;	% this filt will NOT be fit
		Xtarg = 2;
		nim.subunits(end+1) = SUBUNIT(init_filt, weight, NLtype, Xtarg, NLoffset);	% make weight subunit (resets subunits in nim object)
		
		% set stim_params field of NIM object
		stim_params = NIM.create_stim_params([1 Nweight_fits 1]);
		stim_params(2) = NIM.create_stim_params([1 1 1]);
		nim.stim_params = stim_params;
		
	end
	
end

end

